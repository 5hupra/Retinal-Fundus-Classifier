import os
import io
import json
import base64

import cv2
import numpy as np
import torch
import timm
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
WEIGHTS_FILE = os.path.join(MODELS_DIR, "ensemble_weights.json")

CLASS_NAMES = [
    "AMD",
    "Cataract",
    "DR",
    "Glaucoma",
    "Hypertensive Retinopathy",
    "Normal Fundus",
    "Pathological Myopia",
]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_INFO = {
    "AMD":                      {"full": "Age-Related Macular Degeneration", "color": "#e8593c"},
    "Cataract":                 {"full": "Cataract",                         "color": "#f2a623"},
    "DR":                       {"full": "Diabetic Retinopathy",             "color": "#e24b4a"},
    "Glaucoma":                 {"full": "Glaucoma",                         "color": "#7f77dd"},
    "Hypertensive Retinopathy": {"full": "Hypertensive Retinopathy",         "color": "#d4537e"},
    "Normal Fundus":            {"full": "Normal Fundus",                    "color": "#1d9e75"},
    "Pathological Myopia":      {"full": "Pathological Myopia",              "color": "#378add"},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform_224 = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
transform_384 = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
def clahe_crop_resize(img_np: np.ndarray, size: int) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # CLAHE
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Crop — only if result is at least 30% of original area
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        original_area = img_bgr.shape[0] * img_bgr.shape[1]
        if w * h >= 0.30 * original_area:
            img_bgr = img_bgr[y:y+h, x:x+w]

    img_bgr = cv2.resize(img_bgr, (size, size))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ──────────────────────────────────────────────
# Fundus Image Validation
# ──────────────────────────────────────────────
def validate_fundus_image(img_np: np.ndarray) -> tuple:
    """
    Validate if image is a retinal fundus image.
    
    Checks:
    1. Detectable optic disc region (bright circular area)
    2. Color profile (reddish/orange tones characteristic of fundus)
    3. Vessel structures (blood vessels detected via Canny edges)
    4. Size constraints (not too small, not too blurry)
    
    Returns:
        (is_valid, error_message)
    """
    try:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Check 1: Detectable optic disc region
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, "No retinal region detected. Please upload a retinal fundus image."
        
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        original_area = img_bgr.shape[0] * img_bgr.shape[1]
        region_ratio = (w * h) / original_area
        
        if region_ratio < 0.15:
            return (False, 
                    "Image region too small or unclear. "
                    "Please ensure the fundus image is clear and well-cropped.")
        
        # Check 2: Color profile validation - fundus has reddish/orange tones
        roi = img_bgr[y:y+h, x:x+w]
        b, g, r = cv2.split(roi)
        
        r_mean = np.mean(r)
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        
        # Red channel should be higher than blue (fundus characteristic)
        if r_mean <= b_mean:
            return False, "Color profile doesn't match a fundus image."
        
        # Image should not be too dark
        if r_mean < 50:
            return False, "Image is too dark. Please ensure good lighting."
        
        # Check 3: Vessel structure detection (Canny edges for blood vessels)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 30, 100)
        edge_ratio = np.count_nonzero(edges) / edges.size
        
        # Should have detectable blood vessels (1-40% edges)
        if edge_ratio < 0.005:
            return False, "No blood vessel structures detected. Not a fundus image."
        
        if edge_ratio > 0.5:
            return (False, 
                    "Image has excessive noise or texture. "
                    "Please upload a clear fundus photograph.")
        
        # All validation checks passed
        return True, "Valid fundus image"
        
    except Exception as e:
        return False, f"Image validation error: {str(e)}"


# ──────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────
def get_gradcam_target_layer(model):
    """
    Find suitable layer for Grad-CAM.
    
    CRITICAL: Target must output 4D spatial features [batch, channels, H, W].
    ConvNeXt's stages work best with pytorch_grad_cam.
    """
    try:
        # For ConvNeXt: stages have spatial dimensions and work well with GradCAM
        if hasattr(model, 'stages') and len(model.stages) > 0:
            print(f"[Grad-CAM] Target: stages[-1] ({len(model.stages)} stages total)")
            return [model.stages[-1]]
        
        # For EfficientNetV2: try blocks[5] (second-to-last block group)
        if hasattr(model, 'blocks') and len(model.blocks) > 5:
            print(f"[Grad-CAM] Target: blocks[5] (second-to-last block)")
            return [model.blocks[5]]
        
        # For Swin Transformer
        if hasattr(model, 'layers') and len(model.layers) > 0:
            print(f"[Grad-CAM] Target: layers[-1] ({len(model.layers)} layers total)")
            return [model.layers[-1]]
        
        raise ValueError("Could not find suitable layer for Grad-CAM")
        
    except Exception as e:
        print(f"[Grad-CAM] ERROR in get_gradcam_target_layer: {e}")
        raise


def generate_gradcam(
    model: torch.nn.Module,
    img_np: np.ndarray,
    pred_class_idx: int,
    size: int = 384,
):
    """
    Generate Grad-CAM visualization. Returns base64 string or None if fails.
    
    Args:
        model: Model to visualize
        img_np: Image as numpy array (H, W, 3) in RGB
        pred_class_idx: Class index to visualize
        size: Target image size (224 or 384)
    """
    grayscale_cam = None  # Initialize upfront
    
    try:
        print("[Grad-CAM] " + "=" * 50)
        print("[Grad-CAM] Starting visualization...")
        
        # Select appropriate transform based on size
        transform = transform_384 if size >= 384 else transform_224
        
        # Prepare image
        img_float = img_np.astype(np.float32) / 255.0
        transformed = transform(image=img_np)
        tensor = transformed["image"].unsqueeze(0).to(device)
        print(f"[Grad-CAM] Input shape: {tensor.shape}")

        class TargetClass:
            def __init__(self, idx):
                self.idx = idx
            def __call__(self, output):
                # Return simple sum of all activations - safest approach
                # This avoids indexing issues that pytorch_grad_cam may have
                return output.sum()

        # Get target layer
        target_layers = get_gradcam_target_layer(model)
        
        # Ensure eval mode
        model.eval()
        
        # Try GradCAM first, fall back to ScoreCAM if it fails
        try:
            print("[Grad-CAM] Computing GradCAM...")
            with GradCAM(model=model, target_layers=target_layers) as cam:
                result = cam(
                    input_tensor=tensor, 
                    targets=[TargetClass(pred_class_idx)],
                    eigen_smooth=True,
                    aug_smooth=False
                )
                grayscale_cam = result[0]
            
            print(f"[Grad-CAM] GradCAM succeeded! Shape: {grayscale_cam.shape}")
            
        except Exception as grad_e:
            import traceback
            print(f"[Grad-CAM] GradCAM failed with error:")
            traceback.print_exc()
            print(f"[Grad-CAM] Trying ScoreCAM...")
            try:
                from pytorch_grad_cam import ScoreCAM
                print("[Grad-CAM] Computing ScoreCAM...")
                with ScoreCAM(model=model, target_layers=target_layers) as cam:
                    result = cam(
                        input_tensor=tensor,
                        targets=[TargetClass(pred_class_idx)]
                    )
                    grayscale_cam = result[0]
                print(f"[Grad-CAM] ScoreCAM succeeded! Shape: {grayscale_cam.shape}")
            except Exception as score_e:
                print(f"[Grad-CAM] ScoreCAM also failed: {score_e}")
                print("[Grad-CAM] " + "=" * 50)
                return None

        # Verify we got a valid CAM
        if grayscale_cam is None:
            print("[Grad-CAM] CAM is None, returning...")
            print("[Grad-CAM] " + "=" * 50)
            return None
            print("[Grad-CAM] " + "=" * 50)
            return None

        # Create overlay
        print("[Grad-CAM] Creating overlay...")
        overlay = show_cam_on_image(
            img_float, grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
            image_weight=0.45
        )
        
        # Encode
        _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buffer).decode("utf-8")
        
        print(f"[Grad-CAM] Success! Generated {len(b64)} bytes")
        print("[Grad-CAM] " + "=" * 50)
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        print(f"[Grad-CAM] ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("[Grad-CAM] " + "=" * 50)
        return None
        print(traceback.format_exc())  # prints full stack trace
        return None


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
def load_models():
    convnext = timm.create_model(
        "convnext_base.fb_in22k_ft_in1k",
        pretrained=False, num_classes=NUM_CLASSES,
        drop_rate=0.3, drop_path_rate=0.2,
    )
    efficientv2 = timm.create_model(
        "tf_efficientnetv2_m.in21k_ft_in1k",
        pretrained=False, num_classes=NUM_CLASSES,
        drop_rate=0.3, drop_path_rate=0.2,
    )
    swin = timm.create_model(
        "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
        pretrained=False, num_classes=NUM_CLASSES,
        drop_rate=0.3, drop_path_rate=0.2,
    )

    for model, name in [
        (convnext,    "convnext_best.pth"),
        (efficientv2, "efficientv2_best.pth"),
        (swin,        "swin_best.pth"),
    ]:
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Download from Kaggle Output tab and place in backend/models/"
            )
        state = torch.load(path, map_location=device)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        model.to(device)

    with open(WEIGHTS_FILE) as f:
        w = json.load(f)
    weights = np.array(
        [w["convnext"], w["efficientv2"], w["swin"]], dtype=np.float32)

    return convnext, efficientv2, swin, weights






# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(title="Retinal Fundus Classifier", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models...")
try:
    convnext, efficientv2, swin, ensemble_weights = load_models()
    print(f"Models loaded on {device}")
    print(f"Ensemble weights: {ensemble_weights}")
except Exception as e:
    print(f"ERROR loading models: {e}")
    convnext = efficientv2 = swin = ensemble_weights = None


@app.get("/health")
def health():
    return {
        "status":  "ok" if convnext is not None else "models_not_loaded",
        "device":  str(device),
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if convnext is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check backend logs.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image.")

    contents = await file.read()
    try:
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read image file.")

    img_np = np.array(img_pil)

    # Validate that this is actually a fundus image
    is_valid, validation_msg = validate_fundus_image(img_np)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=validation_msg)

    try:
        img_224 = clahe_crop_resize(img_np, 224)
        img_384 = clahe_crop_resize(img_np, 384)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing failed: {str(e)}")

    t_224 = transform_224(image=img_224)["image"].unsqueeze(0).to(device)
    t_384 = transform_384(image=img_384)["image"].unsqueeze(0).to(device)

    # Inference — eval mode, no grad for speed
    with torch.no_grad():
        p1 = torch.softmax(convnext(t_224),    dim=1).cpu().numpy()[0]
        p2 = torch.softmax(efficientv2(t_384), dim=1).cpu().numpy()[0]
        p3 = torch.softmax(swin(t_224),        dim=1).cpu().numpy()[0]

    avg       = (ensemble_weights[0]*p1
               + ensemble_weights[1]*p2
               + ensemble_weights[2]*p3)
    pred_idx  = int(np.argmax(avg))
    pred_name = CLASS_NAMES[pred_idx]
    confidence = float(avg[pred_idx]) * 100

    # Grad-CAM — try ConvNeXt (simpler architecture that works better with GradCAM)
    # Must be outside torch.no_grad() block
    gradcam_b64 = generate_gradcam(convnext, img_224, pred_idx, size=224)

    low_confidence = confidence < 70.0

    scores = {
        CLASS_NAMES[i]: {
            "confidence": round(float(avg[i]) * 100, 2),
            "color":      CLASS_INFO[CLASS_NAMES[i]]["color"],
            "full_name":  CLASS_INFO[CLASS_NAMES[i]]["full"],
        }
        for i in range(NUM_CLASSES)
    }

    return {
        "prediction":     pred_name,
        "confidence":     round(confidence, 2),
        "full_name":      CLASS_INFO[pred_name]["full"],
        "color":          CLASS_INFO[pred_name]["color"],
        "low_confidence": low_confidence,
        "warning":        (
            "Low confidence — please consult an ophthalmologist."
            if low_confidence else None
        ),
        "gradcam":        gradcam_b64,
        "all_scores":     scores,
    }