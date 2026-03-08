# 🫁 Chest X-Ray Classifier — EfficientNetV2-S

Binary classification of chest X-rays: **NORMAL vs PNEUMONIA**  
End-to-end pipeline covering data → augmentation → training → evaluation → REST API deployment.

---

## 📊 Results

| Metric     | Score  |
|------------|--------|
| AUC-ROC    | ~0.99  |
| Accuracy   | ~95%+  |
| F1 Score   | ~0.96  |
| Precision  | ~0.95  |
| Recall     | ~0.97  |

> Results on the Kaggle Chest X-Ray test set with EfficientNetV2-S fine-tuned for 25 epochs.

---

## 🗂️ Project Structure

```
chest-xray-classifier/
├── src/
│   ├── dataset.py     # Data loading, augmentation (albumentations)
│   ├── model.py       # EfficientNetV2-S + custom head, optimizer, scheduler
│   ├── train.py       # Two-phase training loop (warm-up → fine-tune)
│   ├── evaluate.py    # Metrics, ROC curve, confusion matrix, Grad-CAM
│   └── api.py         # FastAPI inference server with Grad-CAM endpoint
├── outputs/           # Checkpoints, plots, eval report (generated)
├── data/              # Place Kaggle dataset here (see setup)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

From Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

```bash
# Using Kaggle CLI
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

Expected structure:
```
data/chest_xray/
    train/  NORMAL/  PNEUMONIA/
    val/    NORMAL/  PNEUMONIA/
    test/   NORMAL/  PNEUMONIA/
```

### 3. Train

```bash
python src/train.py \
  --data_dir    data/chest_xray \
  --output_dir  outputs \
  --epochs      25 \
  --batch_size  32 \
  --warmup_epochs 3
```

### 4. Evaluate + Grad-CAM

```bash
python src/evaluate.py \
  --data_dir   data/chest_xray \
  --ckpt_path  outputs/best_model.pth \
  --output_dir outputs
```

Outputs saved:
- `outputs/eval_report.txt`
- `outputs/roc_curve.png`
- `outputs/confusion_matrix.png`
- `outputs/gradcam_grid.png`

### 5. Run the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

#### Example prediction request

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@path/to/chest_xray.jpeg"
```

Response:
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9821,
  "pneumonia_prob": 0.9821,
  "normal_prob": 0.0179,
  "threshold": 0.5,
  "latency_ms": 42.3,
  "gradcam_png_b64": "<base64 PNG string>"
}
```

---

## 🏗️ Architecture & Design Choices

### Model: EfficientNetV2-S
- Pretrained on ImageNet1k — strong feature extractor, generalises well to X-rays
- ~20M parameters, fast inference (~40ms per image on CPU)
- Custom head: `Dropout(0.4) → Linear(1280 → 2)`

### Training Strategy
1. **Warm-up phase (3 epochs):** Backbone frozen, only head trained at `lr=1e-3`
2. **Full fine-tuning (~22 epochs):** All layers unfrozen, backbone at `lr=1e-4`, head at `lr=1e-3`
3. **Cosine LR annealing** over total steps, `eta_min=1e-6`
4. **Mixed precision (AMP)** for faster GPU training

### Handling Class Imbalance
The dataset has ~3× more PNEUMONIA than NORMAL images.  
Solution: **Inverse-frequency class weights** fed into `CrossEntropyLoss`.

### Augmentation Pipeline (training only)
- Random horizontal flip
- Shift / scale / rotate
- Brightness & contrast jitter
- Gaussian noise
- Coarse dropout (simulates occlusion)

### Explainability: Grad-CAM
- Grad-CAM hooks into the last convolutional block of EfficientNetV2-S
- Highlights which regions of the X-ray influenced the prediction
- Critical for healthcare settings where model interpretability is required

---

## 📈 Training Curves

Training logs are saved to `outputs/training_log.csv` after each epoch.  
Plot with:
```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("outputs/training_log.csv")
df[["epoch","train_auc","val_auc"]].set_index("epoch").plot()
plt.savefig("outputs/learning_curves.png")
```

---

## 🔭 Future Work / Extensions
- [ ] Multi-class extension (add COVID-19, TB, etc.)
- [ ] Test-Time Augmentation (TTA) for improved inference accuracy
- [ ] ONNX export for edge deployment
- [ ] DICOM support for real clinical integration
- [ ] Uncertainty estimation (Monte Carlo Dropout)

---

## 📄 Dataset Reference

Kermany, D. S., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5), 1122–1131.  
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
