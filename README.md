# Retinal Fundus Classifier

7-class retinal disease detection using an ensemble of ConvNeXt-Base,
EfficientNetV2-M and Swin-Small. 95.9% test accuracy.

## Project structure

```
retinal-app/
├── backend/
│   ├── main.py              ← FastAPI server
│   ├── requirements.txt
│   └── models/              ← PUT YOUR .pth FILES HERE
│       ├── convnext_best.pth
│       ├── efficientv2_best.pth
│       ├── swin_best.pth
│       └── ensemble_weights.json
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── main.jsx
    │   └── index.css
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── postcss.config.js
```

---

## Step 1 — Add your model files

Download from Kaggle Output tab and place in `backend/models/`:
- `convnext_best.pth`
- `efficientv2_best.pth`
- `swin_best.pth`
- `ensemble_weights.json`

---

## Step 2 — Start the backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

You should see:
```
Loading models...
Models loaded on cpu   (or cuda if GPU available)
INFO: Uvicorn running on http://127.0.0.1:8000
```

Test it:
```
http://localhost:8000/health
```

---

## Step 3 — Start the frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Usage

1. Drag and drop a retinal fundus image onto the upload zone
2. The app sends it to the backend, runs CLAHE preprocessing + ensemble inference
3. Results show the predicted disease with confidence scores for all 7 classes

---

## Classes

| Class | Description |
|---|---|
| AMD | Age-Related Macular Degeneration |
| Cataract | Lens opacity |
| DR | Diabetic Retinopathy |
| Glaucoma | Optic nerve damage |
| Hypertensive Retinopathy | High blood pressure damage |
| Normal Fundus | No disease |
| Pathological Myopia | Severe nearsightedness |

---

## Note

This is a research tool, not a medical device. Always consult a qualified
ophthalmologist for diagnosis.
