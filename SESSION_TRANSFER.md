# SignalScope — Project Context & Session Transfer Document
**Hackathon:** Smart India Hackathon (SIH 2026) — Problem Statement 2 (SignalScope)  
**Team:** Team Syndicate (L. J. Institute of Engineering and Technology [C-433])  
**Team Members:** Sahil (Lead Developer/Integrator), Aarnav (Team Leader/Model Trainer), Krishna, Het, Jainil, Samina  
**Active GitHub Repository:** https://github.com/AarnavRoy/SIH-2026-with-team-syndicate  

---

## 1. Executive Summary & Architecture Overview

SignalScope is a panoramic media forensics and verification engine designed to detect AI-generated synthetic imagery and provide transparent, multi-signal evidence.

### Implemented Pipeline:
1. **Core AI Model (Section 3.1):**
   - **Architecture:** `EfficientNet-B3` (Aarnav's model, fine-tuned on CIFAKE + GenImage + Midjourney).
   - **Resolution:** 300×300 receptive field.
   - **Inference Purity:** 100% unadulterated raw PyTorch softmax outputs (`fake_prob`, `real_prob`). Zero hardcoded heuristics, zero probability overrides, and zero filename sniffing.
   - **Decision Thresholds:**
     - `fake_prob >= 0.70` $\to$ "Likely AI-Generated"
     - `fake_prob <= 0.35` $\to$ "Likely Authentic Capture"
     - `0.35 < fake_prob < 0.70` $\to$ "Inconclusive / Mixed Signals"

2. **Headline Bonus A — Faithful Explanation (Section 3.2 & 4.3):**
   - **Grad-CAM Feature Heatmaps:** Computed live via PyTorch backward hook on final convolutional layers (`model.features[-1]`), bicubic upsampled, and colorized with custom Jet/Turbo overlay.
   - **Interactive Opacity Control:** Live slider allowing users to fade the heatmap overlay between 0% and 100%.

3. **Headline Bonus D — Hardware Provenance & EXIF (Section 3.2):**
   - Implemented via PIL's native `ExifTags`.
   - Extracts real camera make/model, focal length, ISO, and clean exposure fractions (e.g. `1/14s`).
   - If an image was transferred via WhatsApp / Telegram / Discord / Web, it detects transit-stripping and honestly reports `"No EXIF Found"` rather than fabricating placeholder profiles.

4. **Physics & Frequency Domain — 2D Fourier FFT (Section 6 & 11):**
   - Computes real 2D power spectrum with `numpy.fft.fft2` on the grayscale array.
   - Derives radial energy decay ratio between low-frequency core and high-frequency perimeter.
   - Sets natural optics flag (`decay_ratio >= 0.30`).
   - Detects periodic lattice spikes (deconvolution grid artifacts) while safely ignoring natural high-contrast striped fabrics.

5. **Headline Bonus F — Single-Page Panoramic Studio (Section 3.2):**
   - Fully responsive, 3-column no-scroll studio layout.
   - 4 selectable design themes: **Morning** (warm editorial + 3D bevel), **Linear Dark**, **Editorial Paper**, and **Gemini**.
   - Drag-and-drop file upload, global paste (`Ctrl+V`), and instant preset sample buttons.

6. **Reproducibility & 1-Click Launchers (Section 7.1 & 7.2):**
   - `requirements.txt`: Contains `fastapi`, `uvicorn[standard]`, `python-multipart`, `torch`, `torchvision`, `numpy`, `pillow`.
   - `run.bat`: 1-click Windows launcher that checks dependencies, starts Uvicorn on `http://127.0.0.1:8000`, and automatically opens the browser.
   - `run.sh`: Cross-platform Linux/macOS launcher script.

---

## 2. Important Repository Rules (`app/AGENTS.md`)
1. **Never tweak the backend just for the sake of blind/guessing accuracy.** Predictions must remain authentic to model and sensor telemetry.
2. **Never assume anything on your own.**
3. **Always share a plan and obtain review before executing significant changes.**
4. **Do NOT commit or push large model weights (`*.pth`, `*.zip`) to GitHub.** (Weights are strictly ignored via `.gitignore`).

---

## 3. Pending Open Items for the Next Session

1. **The CMF Phone 1 Multi-Signal Calibration Rule (Bonus D):**
   - *Phenomenon:* When a user uploads an indoor photo taken with a real smartphone (e.g. CMF Phone 1 at `ISO 850`), the phone's aggressive in-camera noise reduction (smoothing) and edge-sharpening fool the CNN into predicting ~84% AI.
   - *Solution:* Per Section 3.2 Bonus D and Section 11, when `is_hardware_camera == True` and `is_natural_optics == True`, the system should cross-reference metadata with the visual model and classify it honestly as **`Inconclusive / Smartphone Post-Processing Detected`** rather than falsely claiming it is an AI generation.
2. **Bonus A Qualitative Cues (Section 4.3):**
   - Current static strings can be dynamically grounded in Grad-CAM peak activations and FFT metrics for maximum faithfulness.
3. **Bonus C Robustness Toggles:**
   - `simulate_jpeg` is fully functional. The remaining two toggles (`Downsampling Defense`, `Screenshot Resampling`) can be wired in `app/vit_service.py` if desired.
4. **Final Deliverables:**
   - Update `README.md` with benchmark citations and setup instructions.
   - Record 3–5 minute demo video.
