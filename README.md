# Retinal Fundus Disease Classification

7-class retinal disease detection using a deep learning ensemble.
Test Accuracy: 95.89% | Macro F1: 95.95%

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/5hupra/Retinal-Fundus-Classifier.git
cd retinal-fundus-classifier
```

### 2. Download the dataset

The dataset is hosted on Kaggle. You need a Kaggle account to download it.

**Balanced dataset (recommended — already preprocessed):**
```
https://www.kaggle.com/datasets/shubhamprakash11/processed-dataset2
```

**Original raw dataset:**
```
https://www.kaggle.com/datasets/shubhamprakash11/retinal-fundus-images
```

Or download via Kaggle CLI:
```bash
pip install kaggle
kaggle datasets download -d shubhamprakash11/processed-dataset2
unzip processed-dataset2.zip -d data/
```

---

### 3. Download the trained models

Models are not included in this repository due to file size (~760 MB total).
Download them from the Kaggle notebook output:
```
https://www.kaggle.com/code/shubhamprakash1751/retinal-fundus-kaggle/output
```

Download these 4 files and place them in `backend/models/`:
```
backend/
└── models/
    ├── convnext_best.pth        (350 MB)
    ├── efficientv2_best.pth     (213 MB)
    ├── swin_best.pth            (196 MB)
    └── ensemble_weights.json    (< 1 KB)
```

---

### 4. Run the backend
```bash
cd backend
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Verify at: `http://localhost:8000/health`

---

### 5. Run the frontend
```bash
cd frontend
npm install
npm run dev
```

Open: `http://localhost:5173`

---

## Model details

| Model | Input | Val F1 |
|---|---|---|
| ConvNeXt-Base | 224px | 97.85% |
| EfficientNetV2-M | 384px | 98.11% |
| Swin-Small | 224px | 97.65% |

Ensemble weights optimised via Nelder-Mead on validation set.

---

## Note

This is a research project, not a certified medical device.
Always consult a qualified ophthalmologist for diagnosis.
```

---

Two things to also add to your `.gitignore` so the model files never accidentally get pushed:
```
# Model weights
backend/models/*.pth
backend/models/*.pt

# Data
data/
*.zip