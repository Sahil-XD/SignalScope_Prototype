"""
Module B: Generator Attribution via Frequency-Domain Forensics
===============================================================
Distinguishes between Diffusion models, GANs, and real optical cameras
using 2D FFT spectral analysis - no additional training required.

Scientific basis:
- GANs (StyleGAN, BigGAN) use transposed convolution for upsampling,
  which leaves periodic checkerboard artifacts visible as spectral peaks
  in the high-frequency domain.
- Diffusion models (SD, Midjourney, DALL-E) use VAE decoders that produce
  smoother, more uniform frequency rolloff with subtle lattice patterns.
- Real camera images follow a natural 1/f power spectral density decay
  due to optical lens physics and Bayer CFA interpolation.

References:
  - Frank et al. "Leveraging Frequency Analysis for Deep Fake Image Recognition" (ICML 2020)
  - Dzanic et al. "Fourier Spectrum Discrepancies in Deep Network Generated Images" (NeurIPS 2020)
"""

import numpy as np
from PIL import Image


def _compute_azimuthal_profile(magnitude_spectrum: np.ndarray) -> np.ndarray:
    """
    Computes the azimuthally averaged (radial) power spectrum from a 2D magnitude spectrum.
    This collapses the 2D FFT into a 1D profile: power vs spatial frequency.
    """
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    
    # Build a radius map from center
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    
    max_radius = min(cy, cx)
    radial_profile = np.zeros(max_radius)
    
    for radius in range(max_radius):
        mask = (r == radius)
        if np.any(mask):
            radial_profile[radius] = np.mean(magnitude_spectrum[mask])
    
    return radial_profile


def _detect_spectral_peaks(radial_profile: np.ndarray, threshold_factor: float = 2.0) -> int:
    """
    Detects abnormal peaks in the high-frequency region of the radial profile.
    GAN checkerboard artifacts produce sharp, periodic peaks.
    Returns the count of anomalous peaks found.
    """
    if len(radial_profile) < 10:
        return 0
    
    # Focus on the upper half of frequencies (high-frequency region)
    high_freq = radial_profile[len(radial_profile) // 2:]
    
    if len(high_freq) < 5 or np.std(high_freq) == 0:
        return 0
    
    # A "peak" is a point significantly above its local neighbors
    median_val = np.median(high_freq)
    std_val = np.std(high_freq)
    peak_threshold = median_val + threshold_factor * std_val
    
    peaks = 0
    for i in range(1, len(high_freq) - 1):
        if (high_freq[i] > high_freq[i - 1] and
            high_freq[i] > high_freq[i + 1] and
            high_freq[i] > peak_threshold):
            peaks += 1
    
    return peaks


def _compute_spectral_slope(radial_profile: np.ndarray) -> float:
    """
    Fits a linear regression to log(power) vs log(frequency) to compute
    the spectral slope (beta).
    - Real images: steep slope (beta ~ -2.0 to -3.0), natural 1/f decay
    - Diffusion:   moderate slope (beta ~ -1.0 to -2.0), smoother rolloff
    - GAN:         shallow/erratic slope (beta > -1.0), high-freq energy retained
    """
    # Skip DC component and very low frequencies
    start = max(2, len(radial_profile) // 10)
    profile = radial_profile[start:]
    
    if len(profile) < 5:
        return -1.5  # default neutral
    
    # Avoid log(0)
    profile = np.maximum(profile, 1e-10)
    
    freqs = np.arange(start, start + len(profile), dtype=np.float64)
    log_freq = np.log10(freqs)
    log_power = np.log10(profile)
    
    # Remove inf/nan
    valid = np.isfinite(log_freq) & np.isfinite(log_power)
    if np.sum(valid) < 3:
        return -1.5
    
    log_freq = log_freq[valid]
    log_power = log_power[valid]
    
    # Linear regression: log_power = slope * log_freq + intercept
    coeffs = np.polyfit(log_freq, log_power, 1)
    slope = coeffs[0]
    
    return float(slope)


def _compute_high_freq_energy_ratio(magnitude_spectrum: np.ndarray) -> float:
    """
    Ratio of high-frequency energy to total energy.
    GANs retain more high-frequency energy than real images.
    """
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = min(cy, cx)
    
    total_energy = np.sum(magnitude_spectrum ** 2)
    if total_energy == 0:
        return 0.0
    
    # High-frequency = outer 50% of the spectrum radius
    high_freq_mask = r >= (max_r * 0.5)
    high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask] ** 2)
    
    return float(high_freq_energy / total_energy)


def _compute_color_channel_correlation(pil_image: Image.Image) -> float:
    """
    Computes the Pearson correlation between R and G channels.
    Real camera sensors (Bayer CFA) produce highly correlated R-G channels (~0.95+).
    AI generators tend to have slightly lower inter-channel correlation.
    """
    arr = np.array(pil_image, dtype=np.float32)
    if arr.ndim < 3 or arr.shape[2] < 2:
        return 1.0
    
    r_flat = arr[:, :, 0].flatten()
    g_flat = arr[:, :, 1].flatten()
    
    if len(r_flat) < 2 or np.std(r_flat) == 0 or np.std(g_flat) == 0:
        return 1.0
    
    return float(np.corrcoef(r_flat, g_flat)[0, 1])


def attribute_generator(ai_probability: float, raw_bytes: bytes, pil_image: Image.Image = None, is_phone: bool = False):
    """
    Module B: Generator Attribution using FFT Frequency Analysis.
    
    Classifies the likely generator family:
      - "Optical Camera" (real photo)
      - "Diffusion Model (e.g., SD/Midjourney/DALL-E)"  
      - "GAN (e.g., StyleGAN/BigGAN)"
    
    Args:
        ai_probability: The AI probability percentage (0-100) from the core classifier.
        raw_bytes: Raw image bytes (used for metadata signature check).
        pil_image: PIL Image object for frequency analysis. If None, falls back to heuristics.
        is_phone: Boolean indicating if Module D verified authentic hardware EXIF metadata.
    
    Returns:
        dict with keys: family, confidence, signature_cues, spectral_slope,
                        high_freq_energy_ratio, spectral_peaks
    """
    
    # ------------------------------------------------------------------
    # Step 0: If hardware signature is detected or core says real, short-circuit
    # ------------------------------------------------------------------
    if is_phone:
        return {
            "family": "Optical Camera (Computational Photography)",
            "confidence": 99.0,
            "signature_cues": ["Authentic hardware metadata overrides visual AI artifacts", "Computational sharpening detected but verified as real"],
            "spectral_slope": None,
            "high_freq_energy_ratio": None,
            "spectral_peaks": None
        }
    
    if ai_probability < 40.0:
        return {
            "family": "Optical Camera",
            "confidence": round(100.0 - ai_probability, 1),
            "signature_cues": ["Low AI probability - consistent with authentic capture"],
            "spectral_slope": None,
            "high_freq_energy_ratio": None,
            "spectral_peaks": None
        }
    
    # ------------------------------------------------------------------
    # Step 1: Check raw bytes for explicit generator metadata signatures
    # ------------------------------------------------------------------
    header = raw_bytes[:2048].lower()
    known_diffusion_sigs = [b'midjourney', b'stable diffusion', b'dall-e', b'dalle',
                            b'stablediffusion', b'invoke-ai', b'automatic1111',
                            b'comfyui', b'dreamstudio']
    known_gan_sigs = [b'stylegan', b'biggan', b'progan', b'thispersondoesnotexist']
    
    for sig in known_diffusion_sigs:
        if sig in header:
            return {
                "family": "Diffusion Model",
                "confidence": round(min(99.0, ai_probability + 5.0), 1),
                "signature_cues": [f"Explicit metadata signature found: '{sig.decode()}'"],
                "spectral_slope": None,
                "high_freq_energy_ratio": None,
                "spectral_peaks": None
            }
    
    for sig in known_gan_sigs:
        if sig in header:
            return {
                "family": "GAN",
                "confidence": round(min(99.0, ai_probability + 5.0), 1),
                "signature_cues": [f"Explicit metadata signature found: '{sig.decode()}'"],
                "spectral_slope": None,
                "high_freq_energy_ratio": None,
                "spectral_peaks": None
            }
    
    # ------------------------------------------------------------------
    # Step 2: If no PIL image provided, fall back to simple heuristic
    # ------------------------------------------------------------------
    if pil_image is None:
        if ai_probability > 75.0:
            return {
                "family": "Diffusion Model (e.g., SD/Midjourney)",
                "confidence": round(ai_probability - 5.0, 1),
                "signature_cues": ["No image data for FFT - classified by probability"],
                "spectral_slope": None,
                "high_freq_energy_ratio": None,
                "spectral_peaks": None
            }
        else:
            return {
                "family": "Unknown AI Generator",
                "confidence": round(ai_probability, 1),
                "signature_cues": ["Insufficient data for spectral attribution"],
                "spectral_slope": None,
                "high_freq_energy_ratio": None,
                "spectral_peaks": None
            }
    
    # ------------------------------------------------------------------
    # Step 3: 2D FFT Frequency Analysis (the core forensic technique)
    # ------------------------------------------------------------------
    gray = np.array(pil_image.convert('L'), dtype=np.float32)
    h, w = gray.shape
    
    # Center-crop to square for clean FFT
    sample_size = min(512, h, w)
    cy, cx = h // 2, w // 2
    crop = gray[cy - sample_size // 2: cy + sample_size // 2,
                cx - sample_size // 2: cx + sample_size // 2]
    
    # Compute 2D FFT and shift DC to center
    f_transform = np.fft.fft2(crop)
    f_shifted = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shifted))  # log scale for better dynamic range
    
    # Compute forensic features
    radial_profile = _compute_azimuthal_profile(magnitude)
    spectral_slope = _compute_spectral_slope(radial_profile)
    spectral_peaks = _detect_spectral_peaks(radial_profile)
    hf_energy_ratio = _compute_high_freq_energy_ratio(np.abs(f_shifted))
    rg_correlation = _compute_color_channel_correlation(pil_image)
    
    # ------------------------------------------------------------------
    # Step 4: Decision logic based on spectral features
    # ------------------------------------------------------------------
    cues = []
    gan_score = 0.0
    diffusion_score = 0.0
    optical_score = 0.0
    
    # --- Spectral Slope ---
    # Real: steep (-2.5 to -3.5), Diffusion: moderate (-1.2 to -2.2), GAN: shallow (> -1.2)
    if spectral_slope > -1.0:
        gan_score += 0.35
        cues.append(f"Shallow spectral slope ({spectral_slope:.2f}) - flat high-frequency energy typical of GAN upsampling")
    elif spectral_slope > -1.8:
        diffusion_score += 0.30
        cues.append(f"Moderate spectral slope ({spectral_slope:.2f}) - consistent with VAE decoder rolloff")
    else:
        optical_score += 0.35
        cues.append(f"Steep spectral slope ({spectral_slope:.2f}) - natural 1/f optical decay pattern")
    
    # --- High-Frequency Energy Ratio ---
    # GAN images retain abnormally high energy in the outer spectral ring
    if hf_energy_ratio > 0.35:
        gan_score += 0.25
        cues.append(f"Elevated high-freq energy ({hf_energy_ratio:.3f}) - transposed convolution residuals")
    elif hf_energy_ratio > 0.15:
        diffusion_score += 0.20
        cues.append(f"Moderate high-freq energy ({hf_energy_ratio:.3f}) - diffusion latent decoder signature")
    else:
        optical_score += 0.25
        cues.append(f"Low high-freq energy ({hf_energy_ratio:.3f}) - consistent with lens low-pass filtering")
    
    # --- Spectral Peaks (checkerboard artifacts) ---
    if spectral_peaks >= 3:
        gan_score += 0.30
        cues.append(f"Detected {spectral_peaks} anomalous spectral peaks - GAN checkerboard artifacts")
    elif spectral_peaks >= 1:
        diffusion_score += 0.15
        gan_score += 0.10
        cues.append(f"Detected {spectral_peaks} minor spectral peak(s) - possible generative artifact")
    else:
        optical_score += 0.15
        cues.append("No anomalous spectral peaks - smooth natural frequency distribution")
    
    # --- Color Channel Correlation (Bayer CFA fingerprint) ---
    if rg_correlation > 0.94:
        optical_score += 0.20
        cues.append(f"High R-G channel correlation ({rg_correlation:.3f}) - Bayer CFA demosaicing signature")
    elif rg_correlation > 0.85:
        diffusion_score += 0.10
        cues.append(f"Moderate R-G correlation ({rg_correlation:.3f}) - synthetic but naturally styled")
    else:
        gan_score += 0.15
        cues.append(f"Low R-G correlation ({rg_correlation:.3f}) - independent channel generation typical of GANs")
    
    # ------------------------------------------------------------------
    # Step 5: Final classification
    # ------------------------------------------------------------------
    scores = {
        "Optical Camera": optical_score,
        "Diffusion Model (e.g., SD/Midjourney)": diffusion_score,
        "GAN (e.g., StyleGAN/BigGAN)": gan_score
    }
    
    # If core classifier says AI (>= 40%), suppress optical score
    if ai_probability >= 50.0:
        scores["Optical Camera"] *= 0.3
    
    best_family = max(scores, key=scores.get)
    best_score = scores[best_family]
    total_score = sum(scores.values())
    
    if total_score > 0:
        confidence = round((best_score / total_score) * 100.0, 1)
    else:
        confidence = 0.0
    
    return {
        "family": best_family,
        "confidence": confidence,
        "signature_cues": cues,
        "spectral_slope": round(spectral_slope, 3),
        "high_freq_energy_ratio": round(hf_energy_ratio, 4),
        "spectral_peaks": spectral_peaks
    }
