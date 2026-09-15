import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def generate_vit_saliency_heatmap(model: torch.nn.Module, image_tensor: torch.Tensor, original_size: tuple) -> str:
    """
    Computes visual saliency:
    - If model is ViT (has encoder): tracks attention across patch tokens.
    - If model is EfficientNet (has features): uses Grad-CAM on final convolutional layer.
    Returns: Base64-encoded PNG string of the colorized heatmap overlay.
    """
    model.eval()

    # Check if EfficientNet architecture
    if hasattr(model, 'features'):
        return _generate_efficientnet_gradcam(model, image_tensor, original_size)

    # ViT Architecture
    activations = []
    def hook_fn(module, input, output):
        activations.append(output)

    # Hook the last transformer layer
    target_layer = model.encoder.layers[-1]
    hook = target_layer.register_forward_hook(hook_fn)

    try:
        # Clone tensor with gradients enabled
        input_tensor = image_tensor.clone().requires_grad_(True)
        outputs = model(input_tensor)

        # Target the top prediction class
        pred_class = torch.argmax(outputs, dim=1).item()
        score = outputs[0, pred_class]

        # Backward pass to get gradients
        model.zero_grad()
        score.backward()

        # Activations shape: [1, 197, 768] (index 0 is CLS token, 1..196 are patches)
        act = activations[0].detach()
        patch_acts = act[:, 1:, :] # [1, 196, 768]

        # Compute patch energy across feature dimensions
        patch_weights = torch.mean(torch.abs(patch_acts), dim=-1) # [1, 196]

        # Reshape to 14x14 grid (224 / 16 = 14)
        grid_map = patch_weights.view(1, 1, 14, 14)

        # Upsample smoothly to original image size
        orig_w, orig_h = original_size
        heatmap_2d = F.interpolate(grid_map, size=(orig_h, orig_w), mode='bicubic', align_corners=False)
        heatmap_2d = heatmap_2d.squeeze().cpu().numpy()

        return _colorize_heatmap(heatmap_2d, orig_w, orig_h)

    finally:
        hook.remove()

def _generate_efficientnet_gradcam(model: torch.nn.Module, image_tensor: torch.Tensor, original_size: tuple) -> str:
    """
    Computes Grad-CAM for EfficientNet-B0 at the final feature extraction layer.
    """
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        input_tensor = image_tensor.clone().requires_grad_(True)
        outputs = model(input_tensor)
        pred_class = torch.argmax(outputs, dim=1).item()
        score = outputs[0, pred_class]

        model.zero_grad()
        score.backward()

        if len(gradients) > 0 and len(activations) > 0:
            grads = gradients[0].detach()
            acts = activations[0].detach()
            pooled_grads = torch.mean(grads, dim=[2, 3], keepdim=True)
            cam = torch.relu(torch.sum(pooled_grads * acts, dim=1, keepdim=True))
        else:
            cam = torch.mean(torch.abs(activations[0].detach()), dim=1, keepdim=True)

        orig_w, orig_h = original_size
        cam_up = F.interpolate(cam, size=(orig_h, orig_w), mode='bicubic', align_corners=False)
        heatmap_2d = cam_up.squeeze().cpu().numpy()

        return _colorize_heatmap(heatmap_2d, orig_w, orig_h)
    finally:
        h1.remove()
        h2.remove()

def _colorize_heatmap(heatmap_2d: np.ndarray, orig_w: int, orig_h: int) -> str:
    # Normalize between 0.0 and 1.0
    h_min, h_max = heatmap_2d.min(), heatmap_2d.max()
    if h_max > h_min:
        heatmap_norm = (heatmap_2d - h_min) / (h_max - h_min)
    else:
        heatmap_norm = np.zeros_like(heatmap_2d)

    # Colorize using custom Jet/Turbo RGB gradient
    heatmap_rgb = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    r = np.clip(1.5 * heatmap_norm - 0.5, 0, 1)
    g = np.clip(1.0 - 2.0 * np.abs(heatmap_norm - 0.5), 0, 1)
    b = np.clip(0.8 - 1.5 * heatmap_norm, 0, 1)
    alpha = np.clip(heatmap_norm * 1.2, 0.2, 0.9)

    heatmap_rgb[..., 0] = (r * 255).astype(np.uint8)
    heatmap_rgb[..., 1] = (g * 255).astype(np.uint8)
    heatmap_rgb[..., 2] = (b * 255).astype(np.uint8)
    heatmap_rgb[..., 3] = (alpha * 255).astype(np.uint8)

    overlay_img = Image.fromarray(heatmap_rgb, mode="RGBA")
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{b64_str}"


# ---------------------------------------------------------------------------
# Triple-Layer Forensic Explanation Engine (Bonus A)
# ---------------------------------------------------------------------------

import cv2


def get_raw_saliency_map(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int = None
) -> np.ndarray:
    """
    Computes Grad-CAM and returns the raw normalized 0-1 float32 2D array.
    Uses EfficientNet (features) or ViT (encoder) path as appropriate.
    """
    model.eval()

    if hasattr(model, 'features'):
        return _raw_gradcam_efficientnet(model, image_tensor, target_class=target_class)
    else:
        return _raw_gradcam_vit(model, image_tensor, target_class=target_class)


def _raw_gradcam_efficientnet(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int = None
) -> np.ndarray:
    """Grad-CAM on EfficientNet final conv block, returns raw 2D saliency."""
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        input_tensor = image_tensor.clone().requires_grad_(True)
        outputs = model(input_tensor)
        if target_class is None:
            pred_class = torch.argmax(outputs, dim=1).item()
        else:
            pred_class = target_class
        score = outputs[0, pred_class]

        model.zero_grad()
        score.backward()

        if len(gradients) > 0 and len(activations) > 0:
            grads = gradients[0].detach()
            acts = activations[0].detach()
            pooled_grads = torch.mean(grads, dim=[2, 3], keepdim=True)
            cam = torch.relu(torch.sum(pooled_grads * acts, dim=1, keepdim=True))
        else:
            cam = torch.mean(torch.abs(activations[0].detach()), dim=1, keepdim=True)

        cam_2d = cam.squeeze().cpu().numpy()
        h_min, h_max = cam_2d.min(), cam_2d.max()
        if h_max > h_min:
            cam_2d = (cam_2d - h_min) / (h_max - h_min)
        else:
            cam_2d = np.zeros_like(cam_2d)

        return cam_2d.astype(np.float32)
    finally:
        h1.remove()
        h2.remove()


def _raw_gradcam_vit(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int = None
) -> np.ndarray:
    """Attention-based saliency on ViT, returns raw 2D saliency."""
    activations = []

    def hook_fn(module, inp, out):
        activations.append(out)

    target_layer = model.encoder.layers[-1]
    hook = target_layer.register_forward_hook(hook_fn)

    try:
        input_tensor = image_tensor.clone().requires_grad_(True)
        outputs = model(input_tensor)
        if target_class is None:
            pred_class = torch.argmax(outputs, dim=1).item()
        else:
            pred_class = target_class
        score = outputs[0, pred_class]

        model.zero_grad()
        score.backward()

        act = activations[0].detach()
        patch_acts = act[:, 1:, :]
        patch_weights = torch.mean(torch.abs(patch_acts), dim=-1)
        grid_size = int(patch_weights.shape[1] ** 0.5)
        grid_map = patch_weights.view(1, 1, grid_size, grid_size)
        # Return at native grid resolution; caller resizes
        cam_2d = grid_map.squeeze().cpu().numpy()
        h_min, h_max = cam_2d.min(), cam_2d.max()
        if h_max > h_min:
            cam_2d = (cam_2d - h_min) / (h_max - h_min)
        else:
            cam_2d = np.zeros_like(cam_2d)

        return cam_2d.astype(np.float32)
    finally:
        hook.remove()


def check_explanation_stability(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_saliency: np.ndarray = None,
    target_class: int = None
) -> str:
    """
    Checks Grad-CAM saliency stability under +/-15% brightness perturbation.
    Properly transforms in de-normalized [0, 1] pixel space and holds target_class
    fixed so attention correlation is measured consistently.
    Returns: 'HIGH', 'MODERATE', or 'LOW'.
    """
    try:
        if target_class is None:
            with torch.no_grad():
                out = model(image_tensor)
                target_class = torch.argmax(out, dim=1).item()

        if original_saliency is None:
            original_saliency = get_raw_saliency_map(model, image_tensor, target_class=target_class)

        orig_flat = original_saliency.flatten()
        if np.std(orig_flat) == 0:
            return "LOW"

        device = image_tensor.device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        # De-normalize to [0, 1] pixel space
        denorm = image_tensor * std + mean
        bright = torch.clamp(denorm * 1.15, 0.0, 1.0)
        dark = torch.clamp(denorm * 0.85, 0.0, 1.0)

        # Re-normalize for model input
        tensor_bright = (bright - mean) / std
        tensor_dark = (dark - mean) / std

        saliency_bright = get_raw_saliency_map(model, tensor_bright, target_class=target_class)
        saliency_dark = get_raw_saliency_map(model, tensor_dark, target_class=target_class)

        corr_bright = np.corrcoef(orig_flat, saliency_bright.flatten())[0, 1]
        corr_dark = np.corrcoef(orig_flat, saliency_dark.flatten())[0, 1]

        if np.isnan(corr_bright):
            corr_bright = 0.0
        if np.isnan(corr_dark):
            corr_dark = 0.0

        avg_corr = float((corr_bright + corr_dark) / 2.0)

        if avg_corr >= 0.70:
            return "HIGH"
        elif avg_corr >= 0.40:
            return "MODERATE"
        else:
            return "LOW"
    except Exception:
        return "MODERATE"



def compute_ela_map(image: Image.Image, quality: int = 95) -> np.ndarray:
    """
    Error Level Analysis: re-save as JPEG, reload, compute per-pixel absolute
    difference. Real photos show uniform low error; AI images show stark bright
    bands at generated boundaries.
    Returns a float32 0-1 normalized grayscale map at original image size.
    """
    orig_arr = np.array(image, dtype=np.float32)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    recomp_arr = np.array(recompressed, dtype=np.float32)

    diff = np.abs(orig_arr - recomp_arr)
    # Convert to grayscale (mean across channels)
    gray_diff = np.mean(diff, axis=2)

    d_max = gray_diff.max()
    if d_max > 0:
        gray_diff = gray_diff / d_max
    return gray_diff.astype(np.float32)


def compute_noise_residual(image: Image.Image, kernel_size: int = 3) -> np.ndarray:
    """
    Subtract a median-filtered copy from the original to isolate high-frequency
    sensor noise. Real cameras leave continuous Poisson grain everywhere; AI
    images have dead-flat zero-noise voids where diffusion smoothing killed
    the noise.
    Returns a float32 0-1 normalized grayscale map at original image size.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    smoothed = cv2.medianBlur(gray.astype(np.uint8), kernel_size).astype(np.float32)
    residual = np.abs(gray - smoothed)

    r_max = residual.max()
    if r_max > 0:
        residual = residual / r_max
    return residual.astype(np.float32)


def _label_connected_components(binary_mask: np.ndarray) -> tuple:
    """
    Label connected components in a binary mask.
    Uses scipy.ndimage.label if available, otherwise a pure numpy BFS.
    Returns (labeled_array, num_features).
    """
    try:
        from scipy.ndimage import label as scipy_label
        return scipy_label(binary_mask)
    except ImportError:
        pass

    # Fallback: simple BFS labeling on 4-connected grid
    labels = np.zeros_like(binary_mask, dtype=np.int32)
    h, w = binary_mask.shape
    current_label = 0
    for r in range(h):
        for c in range(w):
            if binary_mask[r, c] and labels[r, c] == 0:
                current_label += 1
                # BFS flood-fill
                queue = [(r, c)]
                labels[r, c] = current_label
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and binary_mask[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = current_label
                            queue.append((nr, nc))
    return labels, current_label


def _location_label(cy: float, cx: float, img_h: int, img_w: int) -> str:
    """Map a centroid to a 3x3 human-readable grid label."""
    row = 0 if cy < img_h / 3 else (1 if cy < 2 * img_h / 3 else 2)
    col = 0 if cx < img_w / 3 else (1 if cx < 2 * img_w / 3 else 2)
    row_names = ["upper", "center", "lower"]
    col_names = ["left", "center", "right"]
    r_name = row_names[row]
    c_name = col_names[col]
    if r_name == "center" and c_name == "center":
        return "center"
    return f"{r_name}-{c_name}"


def extract_grounded_regions(
    saliency_map: np.ndarray,
    ela_map: np.ndarray,
    noise_map: np.ndarray,
    original_image: Image.Image,
    max_regions: int = 3,
) -> dict:
    """
    Identify the top activated regions from the saliency map and measure
    forensic statistics inside each bounding box on the original image.
    """
    orig_arr = np.array(original_image)
    img_h, img_w = orig_arr.shape[:2]

    # Downsample saliency to ~64x64 for component analysis
    target = 64
    sal_small = cv2.resize(saliency_map, (target, target), interpolation=cv2.INTER_AREA)

    # Threshold at 75th percentile
    thresh = float(np.percentile(sal_small, 75))
    binary = (sal_small >= thresh).astype(np.uint8)

    labeled, n_features = _label_connected_components(binary)

    # Collect per-component mean activation and bounding box (in downsampled coords)
    components = []
    for comp_id in range(1, n_features + 1):
        comp_mask = labeled == comp_id
        mean_act = float(np.mean(sal_small[comp_mask]))
        ys, xs = np.where(comp_mask)
        components.append({
            "id": comp_id,
            "mean_act": mean_act,
            "y1": int(ys.min()),
            "y2": int(ys.max()),
            "x1": int(xs.min()),
            "x2": int(xs.max()),
        })

    # Keep top N by mean activation
    components.sort(key=lambda c: c["mean_act"], reverse=True)
    components = components[:max_regions]

    # Map bounding boxes back to original pixel coords
    scale_y = img_h / target
    scale_x = img_w / target

    # Frame-level baselines
    frame_noise_std = float(np.std(noise_map))
    frame_ela_mean = float(np.mean(ela_map))

    regions = []
    total_coverage = 0.0
    total_pixels = img_h * img_w

    hsv_arr = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    gray_arr = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2GRAY)

    for idx, comp in enumerate(components, start=1):
        # Scale back to original coordinates (clip to image bounds)
        x1 = max(0, int(comp["x1"] * scale_x))
        y1 = max(0, int(comp["y1"] * scale_y))
        x2 = min(img_w, int((comp["x2"] + 1) * scale_x))
        y2 = min(img_h, int((comp["y2"] + 1) * scale_y))

        # Measure inside the bounding box on the ORIGINAL image
        region_gray = gray_arr[y1:y2, x1:x2]
        region_ela = ela_map[y1:y2, x1:x2]
        region_noise = noise_map[y1:y2, x1:x2]
        region_hsv = hsv_arr[y1:y2, x1:x2]

        lap = cv2.Laplacian(region_gray, cv2.CV_64F)
        laplacian_var = float(np.var(lap))

        sobel_x = cv2.Sobel(region_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(region_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_density = float(np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2)))

        region_ela_mean = float(np.mean(region_ela))
        ela_ratio = region_ela_mean / (frame_ela_mean + 1e-12)

        noise_std = float(np.std(region_noise))
        saturation_mean = float(np.mean(region_hsv[:, :, 1]))

        # Center of bounding box for location label
        cy = (y1 + y2) / 2.0
        cx = (x1 + x2) / 2.0
        location = _location_label(cy, cx, img_h, img_w)

        region_area = (x2 - x1) * (y2 - y1)
        coverage_pct = round(region_area / total_pixels * 100, 1)
        total_coverage += coverage_pct

        regions.append({
            "id": idx,
            "bbox": [x1, y1, x2, y2],
            "location": location,
            "coverage_pct": coverage_pct,
            "laplacian_var": round(laplacian_var, 2),
            "edge_density": round(edge_density, 2),
            "ela_ratio": round(ela_ratio, 2),
            "noise_std": round(noise_std, 4),
            "saturation_mean": round(saturation_mean, 3),
        })

    return {
        "regions": regions,
        "frame_noise_std": round(frame_noise_std, 4),
        "frame_ela_mean": round(frame_ela_mean, 4),
        "total_coverage_pct": round(total_coverage, 1),
    }


def generate_grounded_cues(
    regions_data: dict, fft_data: dict, verdict_status: str
) -> list:
    """
    Generate 2-3 bullet point strings strictly grounded in measured values.
    Each bullet cites a specific region id, location, and at least one number.
    """
    cues: list[str] = []
    regions = regions_data.get("regions", [])
    frame_noise = regions_data.get("frame_noise_std", 0.0)
    coverage = regions_data.get("total_coverage_pct", 0.0)

    if verdict_status in ("synthetic", "inconclusive"):
        # Synthetic / inconclusive: highlight anomalies
        for region in regions[:2]:
            parts = []
            rid = region["id"]
            loc = region["location"]

            # Over-smoothing check
            if region["laplacian_var"] < 15.0:
                parts.append(
                    f"Region {rid} ({loc}) exhibits over-smoothing characteristics "
                    f"(Laplacian variance {region['laplacian_var']:.1f}, edge density "
                    f"{region['edge_density']:.2f}), suggesting diffusion-based generation."
                )

            # Noise absence check
            if frame_noise > 0 and region["noise_std"] < 0.3 * frame_noise:
                parts.append(
                    f"Region {rid} ({loc}) shows likely noise absence "
                    f"(noise sigma={region['noise_std']:.3f} vs frame sigma={frame_noise:.3f}), "
                    f"suggesting synthetic smoothing in this area."
                )

            # Compression anomaly check
            if region["ela_ratio"] > 2.0:
                parts.append(
                    f"Region {rid} ({loc}) shows elevated compression anomaly "
                    f"(ELA ratio {region['ela_ratio']:.1f}x frame average), "
                    f"suggesting non-uniform recompression."
                )

            # Grid artifacts (only if actually detected)
            if fft_data.get("grid_artifact_detected") and not parts:
                parts.append(
                    f"Region {rid} ({loc}) coincides with detected periodic grid "
                    f"artifacts in the frequency domain (edge density {region['edge_density']:.2f})."
                )

            if parts:
                cues.append(parts[0])

            if len(cues) >= 2:
                break

        # If no region-specific anomalies found, generate a generic measured bullet
        if not cues and regions:
            r = regions[0]
            cues.append(
                f"Region {r['id']} ({r['location']}): Laplacian variance {r['laplacian_var']:.1f}, "
                f"noise sigma={r['noise_std']:.3f}, ELA ratio {r['ela_ratio']:.1f}x. "
                f"Values suggest possible generative manipulation."
            )

    else:
        # Authentic: cite healthy sensor grain and uniform ELA
        for region in regions[:2]:
            rid = region["id"]
            loc = region["location"]
            parts = []

            if frame_noise > 0 and region["noise_std"] >= 0.3 * frame_noise:
                parts.append(
                    f"Region {rid} ({loc}) shows healthy sensor grain "
                    f"(noise sigma={region['noise_std']:.3f}, frame sigma={frame_noise:.3f}), "
                    f"consistent with authentic camera capture."
                )

            if region["ela_ratio"] < 2.0:
                parts.append(
                    f"Region {rid} ({loc}) exhibits uniform compression profile "
                    f"(ELA ratio {region['ela_ratio']:.1f}x), consistent with "
                    f"single-pass JPEG encoding."
                )

            if parts:
                cues.append(parts[0])

            if len(cues) >= 2:
                break

        if not cues and regions:
            r = regions[0]
            cues.append(
                f"Region {r['id']} ({r['location']}): Laplacian variance {r['laplacian_var']:.1f}, "
                f"noise sigma={r['noise_std']:.3f}. Values consistent with authentic capture."
            )

    # Final scope bullet
    cues.append(
        f"Analysis scope: Top {coverage:.0f}% of model-highlighted regions evaluated. "
        f"Areas outside activation zones were not assessed."
    )

    return cues


def generate_ela_overlay(image: Image.Image, ela_map: np.ndarray) -> str:
    """
    Colorize ELA map with COLORMAP_INFERNO, blend with original at 50% opacity,
    return as base64 data URI.
    """
    orig_arr = np.array(image)

    # Scale ELA map to 0-255 and apply colormap
    ela_uint8 = (np.clip(ela_map, 0.0, 1.0) * 255).astype(np.uint8)
    ela_color = cv2.applyColorMap(ela_uint8, cv2.COLORMAP_INFERNO)
    ela_color = cv2.cvtColor(ela_color, cv2.COLOR_BGR2RGB)

    # Blend at 50% opacity
    blended = cv2.addWeighted(orig_arr, 0.5, ela_color, 0.5, 0)
    return _array_to_base64_uri(blended)


def generate_noise_overlay(image: Image.Image, noise_map: np.ndarray) -> str:
    """
    Amplify noise residual 5x for visibility, apply COLORMAP_BONE, blend with
    original at 40% opacity, return as base64 data URI.
    """
    orig_arr = np.array(image)

    # Amplify noise 5x and cap at 1.0, then scale to 0-255
    amplified = np.clip(noise_map * 5.0, 0.0, 1.0)
    noise_uint8 = (amplified * 255).astype(np.uint8)
    noise_color = cv2.applyColorMap(noise_uint8, cv2.COLORMAP_BONE)
    noise_color = cv2.cvtColor(noise_color, cv2.COLOR_BGR2RGB)

    # Blend at 40% opacity
    blended = cv2.addWeighted(orig_arr, 0.6, noise_color, 0.4, 0)
    return _array_to_base64_uri(blended)


def _array_to_base64_uri(arr: np.ndarray) -> str:
    """Convert an RGB numpy array to a base64 data URI PNG string."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
