# SignalScope — Bonus Modules Improvement Plan

**Team Syndicate** | SIH 2026 | Problem Statement 2  
*Generated: 15 Sept 2026*

---

## Priority Ranking (Impact vs Effort)

| Priority | Module | Effort | Score Impact |
|---|---|---|---|
| 🔴 1st | **A — Fix Explanation Cues** | Low | High (15 pts axis) |
| 🟡 2nd | **C — Degradation Sweep** | Low | Medium |
| 🟢 3rd | **E — Add CLIP Multimodal** | Medium | Medium (differentiator) |
| 🔵 4th | **B — Generator Attribution** | Medium-High | Medium |
| ⚪ 5th | **D — Polish Provenance** | Low | Low (already strong) |

---

## Module A — Faithful Explanation (Improve Existing)

### Current State
- Grad-CAM heatmap works (`app/explain_service.py`)
- Text cues in `app/vit_service.py` (L246-290) are **hardcoded template strings**, not grounded in actual image analysis

### Problem
PDF Section 4.3 scores on **faithfulness, not fluency**. Judges verify if cited cues correspond to *real* artifacts. Current cues like *"Latent checkerboard grid identified"* are overclaiming if no checkerboard was actually detected.

### Improvements Needed
1. **Ground cues in Grad-CAM output** — analyze which image regions the heatmap activates on (face/hands/text/background) and report that specifically
2. **Add artifact-specific detection** — check for common AI tells:
   - Unnatural finger counts
   - Warped/inconsistent text
   - Inconsistent lighting/shadows
   - Over-smooth skin texture
   - Even simple heuristics (e.g., edge density in high-activation regions) score higher than templates
3. **Honest hedging** — use "likely", never "certain" per PDF scope rules

### Key Files to Modify
- `app/vit_service.py` — replace template cues with grounded analysis
- `app/explain_service.py` — extract region-level activation stats from Grad-CAM

---

## Module B — Generator Attribution (New)

### Current State
- Not implemented. System only does binary real/fake.

### What's Needed
Predict the **generator family** (GAN vs Diffusion vs specific model) with a separate reported metric.

### Implementation Options

**Option 1 — Multi-class head (best):**
- Change classifier head from 2-class to multi-class (Real / SD / Midjourney / DALL-E / GAN)
- Use GenImage dataset — already organized by generator, labels are free
- Report multi-class macro-F1

**Option 2 — FFT heuristic (faster):**
- GANs produce periodic grid artifacts in the frequency spectrum
- Diffusion models show different high-frequency energy signatures
- Use existing `analyze_frequency_domain()` output to classify generator family
- Lower accuracy but zero retraining

### Key Files to Modify
- `app/vit_service.py` — add attribution logic and output field
- `model/train.py` — retrain with multi-class labels (Option 1)
- `app/static/index.html` — display attribution result in UI

---

## Module C — Robustness to Degradation (Improve Existing)

### Current State
- Single JPEG q=50 stress toggle exists
- No systematic degradation-vs-accuracy analysis as PDF requires

### Improvements Needed

1. **Sweep multiple degradation levels** — not just q=50:
   - JPEG quality: q=90, 70, 50, 30, 10
   - Resize: 75%, 50%, 25%
   - Screenshot simulation: re-encode as PNG after JPEG

2. **Return structured analysis** from the API:
   ```json
   {
     "robustness_sweep": [
       {"degradation": "Original", "ai_prob": 92.3, "delta": 0.0},
       {"degradation": "JPEG q=70", "ai_prob": 91.1, "delta": -1.2},
       {"degradation": "JPEG q=50", "ai_prob": 88.7, "delta": -3.6},
       {"degradation": "Resize 50%", "ai_prob": 85.2, "delta": -7.1}
     ]
   }
   ```

3. **Display chart/table in UI** showing confidence stability across degradations

### Key Files to Modify
- `app/vit_service.py` — add `run_robustness_sweep()` function
- `app/main.py` — add `/api/robustness` endpoint or extend `/api/analyze`
- `app/static/index.html` — render degradation chart in Robustness Suite tab

---

## Module D — Provenance & Metadata (Polish Existing)

### Current State
- EXIF extraction (`app/exif_service.py`) — solid
- FFT analysis (`app/vit_service.py`) — solid
- Multi-signal calibration logic — solid

### Improvements Needed

1. **Add C2PA / Content Credentials parsing** — PDF specifically mentions this
   ```bash
   pip install c2pa-python
   ```
   Even just checking for C2PA manifest presence and reporting it adds points.

2. **Render FFT spectrum visually** — save the 2D power spectrum as a heatmap image and display in UI. Already computed in `analyze_frequency_domain()`, just needs visualization.

3. **Better calibration explanation** — when calibration fires, be more specific:
   > "ISO 1600 at f/1.8 triggers heavy MediaTek ISP noise reduction, which mimics diffusion smoothing patterns."

### Key Files to Modify
- `app/exif_service.py` — add C2PA manifest check
- `app/vit_service.py` — return FFT spectrum as base64 image
- `app/static/index.html` — display FFT visualization and C2PA status

---

## Module E — Multimodal Image+Text (New)

### Current State
- Not implemented.

### What's Needed
Given an image + a caption, assess if the text matches the image content.

### Implementation (CLIP-based, simplest)

```python
# New file: app/clip_service.py
import clip
import torch
from PIL import Image

model, preprocess = clip.load("ViT-B/32")

def check_caption_consistency(image_path: str, caption: str) -> dict:
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text = clip.tokenize([caption])
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = (image_features @ text_features.T).item()
    
    return {
        "caption": caption,
        "similarity_score": round(similarity, 3),
        "consistency": "High" if similarity > 0.25 else "Low",
        "note": "Caption matches image content" if similarity > 0.25 
               else "Caption may not match image content"
    }
```

### Integration Points
- Add `--caption` flag to `model/predict.py`
- Add caption input field in `app/static/index.html`
- Add `caption` form field to `/api/analyze` in `app/main.py`
- **PDF caveat:** text is generic captions, NOT political claims. Frame as consistency check.

### Key Files to Create/Modify
- `app/clip_service.py` — **[NEW]** CLIP consistency checker
- `app/main.py` — accept optional caption parameter
- `model/predict.py` — add `--caption` CLI flag
- `requirements.txt` — add `clip` dependency
- `app/static/index.html` — add caption input field

---

## Quick Reference: PDF Scoring Axes

| Parameter | Weight | What Matters |
|---|---|---|
| AI/ML Implementation | 25 | Unseen-generator AUC (primary), honest metrics |
| Technical Implementation | 20 | Reproducibility, code quality, deployment |
| Innovation & Creativity | 15 | Novel approaches, strong generalisation ideas |
| **Explanation & Trust** | **15** | **Faithfulness rubric (Module A) — biggest bonus win** |
| User Experience | 10 | Clear verdict, responsible language, usability |
| Problem Understanding | 10 | Generalisation grasp, honest limitations |
| Presentation & Demo | 5 | 3-5 min video clarity |

---

## Gemini API Integration (Modules A & E)

### Setup

```bash
pip install google-generativeai python-dotenv
```

```env
# .env (ADD TO .gitignore!)
GEMINI_API_KEY=AIzaSy...
```

```python
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")
```

### Module A — Gemini-Enhanced Explanation

Send image + Grad-CAM heatmap to Gemini Vision for grounded artifact descriptions:

```python
import PIL.Image
img = PIL.Image.open(image_path)
response = model.generate_content([
    img,
    "Analyze this image for signs of AI generation. "
    "List specific visual artifacts you observe (texture, geometry, lighting, anatomy). "
    "Use 'likely' not 'certain'. Be concise."
])
cues = response.text
```

### Module E — Gemini Image-Caption Consistency

```python
response = model.generate_content([
    img,
    f'Does this caption accurately describe this image? Caption: "{caption}". '
    f'Rate consistency 0-100 and explain mismatches.'
])
```

### Risk Assessment

| Risk | Why It Matters |
|---|---|
| **Faithfulness** | Gemini can hallucinate artifacts — PDF rubric 4.3 penalizes "fluent but wrong" |
| **API dependency** | If API is down during judging, module breaks |
| **Latency** | Each call adds 2-5 seconds |
| **"Outsourcing" perception** | Judges may see it as delegating reasoning to a third-party LLM |
| **Rate limits** | Free tier may throttle during demo day |

### Recommended Hybrid Architecture

Use local methods as primary (works offline for judges), Gemini as optional enhancement:

| Module | Primary (Local, No API) | Secondary (Gemini, Optional) |
|---|---|---|
| **A** | Grad-CAM + local heuristics (edge density, smoothness in activated regions) | Gemini describes high-activation regions for richer text |
| **E** | CLIP similarity score (fast, offline, reliable) | Gemini for detailed mismatch explanation when CLIP flags low similarity |

### Alternative: Gemma (Local, No API Key)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it")
```

- **Pros:** Offline, free, no rate limits, fully reproducible without API key
- **Cons:** ~8GB RAM, slower, less capable than hosted Gemini
- **Verdict:** Good fallback; CLIP is simpler for Module E
