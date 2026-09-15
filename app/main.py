from pathlib import Path
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.vit_service import analyze_image

app = FastAPI(
    title="SignalScope Forensic Engine",
    description="Media Authenticity & Interpretability Verification Engine",
    version="2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Serve static web frontend
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "SignalScope Engine Online", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "active_model": "EfficientNet-B3 (CIFAKE + GenImage + Midjourney)",
        "available_models": ["efficientnet_b3"],
        "benchmark_auc": 0.9942,
        "input_resolution": "300x300",
        "parameter_count": "12.2M"
    }

@app.post("/api/analyze")
@app.post("/predict")
async def analyze_endpoint(
    file: UploadFile = File(...),
    simulate_jpeg: bool = Form(False),
    model_type: str = Form("efficientnet_b3"),
    caption: str = Form(None)
):
    try:
        # Save temporary uploaded file
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            # Run multi-signal forensic evaluation pipeline
            results = analyze_image(tmp_path, simulate_jpeg=simulate_jpeg, model_type=model_type, caption=caption)
            results["filename"] = file.filename
            return JSONResponse(content=results)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/analyze-sample")
async def analyze_sample_endpoint(
    sample_id: str = Form(...),
    simulate_jpeg: bool = Form(False),
    model_type: str = Form("efficientnet_b3")
):
    try:
        import base64
        base_dir = Path(__file__).resolve().parent.parent
        if sample_id == "backpack":
            sample_path = base_dir / "data" / "real_object_image.jpeg"
            filename = "real_object_image.jpeg"
        elif sample_id == "dalle":
            sample_path = base_dir / "first" / "ChatGPT Image Sep 11, 2026, 11_04_46 PM.png"
            filename = "dalle3_santorini_cat.png"
        elif sample_id == "lake":
            sample_path = base_dir / "first" / "ChatGPT Image Sep 11, 2026, 10_59_50 PM.png"
            filename = "dalle3_lake_reflection.png"
        else:
            sample_path = base_dir / "data" / "real_object_image.jpeg"
            filename = "real_object_image.jpeg"

        if not sample_path.exists():
            return JSONResponse(status_code=404, content={"success": False, "error": f"Sample file {sample_path} not found"})

        with open(sample_path, "rb") as f:
            orig_b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = sample_path.suffix.lower().replace(".", "")
        if ext == "jpg":
            ext = "jpeg"
        orig_data_url = f"data:image/{ext};base64,{orig_b64}"

        results = analyze_image(str(sample_path), simulate_jpeg=simulate_jpeg, model_type=model_type)
        results["filename"] = filename
        results["image_data_url"] = orig_data_url
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
