import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io

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
    "Hypertensive Retinopathy": {"full": "Hypertensive Retinopathy",        "color": "#d4537e"},
    "Normal Fundus":            {"full": "Normal Fundus",                    "color": "#1d9e75"},
    "Pathological Myopia":      {"full": "Pathological Myopia",             "color": "#378add"},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Preprocessing
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

def clahe_crop_resize(img_np: np.ndarray, size: int) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab     = cv2.merge((clahe.apply(l), a, b))
    img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img_bgr = img_bgr[y:y+h, x:x+w]

    img_bgr = cv2.resize(img_bgr, (size, size))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

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
                f"Download your .pth files from Kaggle Output tab "
                f"and place them in backend/models/"
            )
        state = torch.load(path, map_location=device)
        # Handle DataParallel wrapped models
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        model.to(device)

    with open(WEIGHTS_FILE) as f:
        w = json.load(f)
    weights = np.array([w["convnext"], w["efficientv2"], w["swin"]], dtype=np.float32)

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

# Load models at startup
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
        raise HTTPException(status_code=503,
                            detail="Models not loaded. Check backend logs.")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
                            detail="File must be an image.")

    # Read image
    contents = await file.read()
    img_pil  = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np   = np.array(img_pil)

    # Preprocess at both resolutions
    img_224 = clahe_crop_resize(img_np, 224)
    img_384 = clahe_crop_resize(img_np, 384)

    t_224 = transform_224(image=img_224)["image"].unsqueeze(0).to(device)
    t_384 = transform_384(image=img_384)["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        p1 = torch.softmax(convnext(t_224),    dim=1).cpu().numpy()[0]
        p2 = torch.softmax(efficientv2(t_384), dim=1).cpu().numpy()[0]
        p3 = torch.softmax(swin(t_224),        dim=1).cpu().numpy()[0]

    avg       = (ensemble_weights[0]*p1
               + ensemble_weights[1]*p2
               + ensemble_weights[2]*p3)
    pred_idx  = int(np.argmax(avg))
    pred_name = CLASS_NAMES[pred_idx]

    scores = {
        CLASS_NAMES[i]: {
            "confidence": round(float(avg[i]) * 100, 2),
            "color":      CLASS_INFO[CLASS_NAMES[i]]["color"],
            "full_name":  CLASS_INFO[CLASS_NAMES[i]]["full"],
        }
        for i in range(NUM_CLASSES)
    }

    return {
        "prediction":  pred_name,
        "confidence":  round(float(avg[pred_idx]) * 100, 2),
        "full_name":   CLASS_INFO[pred_name]["full"],
        "color":       CLASS_INFO[pred_name]["color"],
        "all_scores":  scores,
    }
