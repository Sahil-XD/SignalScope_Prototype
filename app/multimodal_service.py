"""
Module E: Multimodal Image-Text Verification

Compares the user-provided caption against the visual content of the image.
Uses OpenAI's CLIP model (if transformers is installed) to compute cosine similarity
between the image embeddings and text embeddings.
"""

import warnings

def verify_caption_consistency(pil_image, caption: str) -> dict:
    """
    Checks if the provided caption is consistent with the image.
    Returns a dictionary with 'is_consistent', 'confidence', and 'note'.
    """
    if not caption or not caption.strip():
        return {
            "provided": False,
            "is_consistent": None,
            "confidence": 0.0,
            "note": "No caption provided for multimodal verification."
        }
        
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        import torch.nn.functional as F
        
        # Load lightweight CLIP model
        # Using a try/except to avoid downloading multiple times or crashing
        model_id = "openai/clip-vit-base-patch32"
        
        # In a real deployed scenario, this model should be loaded once globally in app.py.
        # For the hackathon demo, we load it here, but it will be slow on the first run.
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
        
        inputs = processor(text=[caption], images=pil_image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Image-text similarity score (cosine similarity * 100)
        logits_per_image = outputs.logits_per_image 
        similarity_score = logits_per_image.item() # usually between 10 and 35
        
        # Normalize arbitrarily for UI confidence (CLIP logits are uncalibrated)
        # 25+ is usually a strong match. < 20 is usually a mismatch.
        normalized_score = max(0.0, min(100.0, (similarity_score - 15) * 5))
        
        is_consistent = similarity_score > 22.0
        
        if is_consistent:
            note = f"Context Verified: The caption strongly matches the visual content."
        else:
            note = f"Context Mismatch: The caption does not appear to describe the visual content."
            
        return {
            "provided": True,
            "is_consistent": is_consistent,
            "confidence": round(normalized_score, 1),
            "note": note,
            "raw_score": round(similarity_score, 2)
        }
        
    except ImportError:
        # Fallback if transformers isn't installed
        return {
            "provided": True,
            "is_consistent": True, # Assume true in fallback to not penalize
            "confidence": 50.0,
            "note": f"Caption received ('{caption}'). (Install 'transformers' to enable active CLIP verification)."
        }
    except Exception as e:
        return {
            "provided": True,
            "is_consistent": None,
            "confidence": 0.0,
            "note": f"Verification failed: {str(e)}"
        }
