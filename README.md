# 🪖 Helmet Compliance Detection using YOLOv8

## 📌 Project Overview

This project implements a real-time helmet compliance detection system using YOLOv8 (Ultralytics).  
The model detects whether a person is wearing a helmet or not using deep learning-based object detection.

The system was trained using transfer learning and evaluated using standard object detection metrics including mAP, precision, and recall. Real-time inference is supported via webcam integration.

---

## 🧠 Model Architecture

- **Model:** YOLOv8 (Ultralytics)
- **Framework:** PyTorch
- **Pretraining:** COCO dataset (Transfer Learning)
- **Input Resolution:** 640 × 640
- **Epochs:** 50
- **Classes:**
  - With Helmet
  - Without Helmet

YOLOv8 is a one-stage object detector that performs:
- Bounding box regression
- Object classification
- Non-Max Suppression (NMS)

---

## 📂 Dataset

The project uses the **Helmet Detection Dataset** from Kaggle:

https://www.kaggle.com/datasets/andrewmvd/helmet-detection

### Dataset Processing

- Images sourced from Kaggle
- Annotated and converted using **Roboflow**
- Exported in YOLOv8 format
- Train/Validation/Test split:
  - 70% Training
  - 20% Validation
  - 10% Testing
- Total Images Used: 300+

Classes:
- With Helmet
- Without Helmet

---

## 🚀 Training

To train the model from scratch using YOLOv8:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### Training Details

- Optimizer: Default YOLOv8 optimizer (SGD-based with momentum)
- Loss Components:
  - Box Loss
  - Classification Loss
  - Distribution Focal Loss (DFL)
- Transfer learning from pretrained COCO weights

Training outputs are saved inside:

```
runs/detect/train/
```

Best model weights:

```
runs/detect/train/weights/best.pt
```

---

## 📊 Performance Metrics

- **Precision:** 0.63
- **Recall:** 0.79
- **mAP@0.5:** 0.75
- **mAP@0.5:0.95:** 0.45

### Per-Class Performance

**With Helmet**
- mAP@0.5: 0.87
- Recall: 0.89

**Without Helmet**
- mAP@0.5: 0.62
- Recall: 0.69

Performance variation is primarily due to class imbalance and challenging visibility conditions.

---

## ⚡ Inference Performance

- GPU Inference Time: ~3.3 ms per image
- Real-time webcam detection supported
- Model Variant: YOLOv8n (lightweight)

---

## ▶️ Running Inference

### Webcam Detection

```bash
yolo detect predict model=weights/best.pt source=0
```

### Image Detection

```bash
yolo detect predict model=weights/best.pt source=image.jpg
```

Predictions are saved inside:

```
runs/detect/predict/
```

---

## 🛠 Installation

### Clone Repository

```bash
git clone <https://github.com/Aditya-0510/helmet_detection>
cd helmet-detection-yolov8
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Concepts Applied

- Object Detection
- Transfer Learning
- Intersection over Union (IoU)
- Non-Max Suppression (NMS)
- Precision–Recall Tradeoff
- Mean Average Precision (mAP)
- Bounding Box Regression
- Class Imbalance Handling

---

## 🔮 Future Improvements

- Train using YOLOv8s or YOLOv8m for improved accuracy
- Increase dataset size for better generalization
- Deploy as REST API
- Edge deployment (Jetson Nano / Raspberry Pi)
- Helmet violation counting system

---

## 👨‍💻 Author

Sai Aditya  