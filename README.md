# 🚗 Adaptive Cruise Control — ML-Based Perception System

An end-to-end machine learning pipeline for real-time scene understanding in adaptive cruise control systems. The model performs semantic segmentation across 13 object classes critical to autonomous driving, enabling vehicles to perceive and respond to their environment accurately and safely.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Final Validation Pixel Accuracy | **90.66%** |
| Final Validation mIoU | **35.72%** |

### Per-Class IoU Breakdown

| Class | IoU |
|---|---|
| Road | 91.14% |
| Unlabeled | 93.91% |
| Building | 71.09% |
| Car | 57.00% |
| Sidewalk | 58.78% |
| Pedestrian | 45.62% |
| Road Line | 27.68% |
| Wall | 17.45% |
| Vegetation | 1.28% |
| Pole | 0.26% |
| Other | 0.12% |
| Traffic Sign | 0.00% |
| Fence | 0.00% |

> The model performs strongly on safety-critical classes such as **Road (91.14%)** and **Unlabeled background (93.91%)**, which are the most important for cruise control decision-making.

---

## 🎯 Region of Interest (ROI)

To improve model efficiency and focus predictions on relevant areas, a **Region of Interest** is applied to each frame before segmentation. This crops or masks the input image to concentrate on the road ahead — the area most critical for cruise control decisions.

- **Purpose** — Eliminates irrelevant pixels (sky, distant background) that add noise without contributing to driving decisions
- **Effect on accuracy** — Focusing the model on the lower portion of the frame improves performance on key classes like Road (91.14%) and Road Line
- **Implementation** — Defined in `segmentation_breaking.py` as part of the preprocessing pipeline, applied uniformly across all 27 video sequences before training

---

## 🧠 How It Works

The system uses a semantic segmentation model trained on driving scene data, classifying every pixel in a camera frame into one of 13 categories. This pixel-level understanding feeds into the cruise control logic to:

1. Detect and track vehicles ahead
2. Identify road boundaries and lane lines
3. Recognise pedestrians and obstacles
4. Adjust speed dynamically based on scene context

---

## 🛠️ Getting Started

```bash
# Clone the repository
git clone https://github.com/yourname/adaptive-cruise-control.git
cd adaptive-cruise-control

# Install dependencies
pip install -r requirements.txt

# Preprocess dataset and run segmentation pipeline
python segmentation_breaking.py

# Train and evaluate the model
python modelling.py
```

---

## 📁 Project Structure

```
adaptive-cruise-control/
├── dataset/
│   ├── images/                  # Raw image data
│   │   ├── video_001/           # 27 folders of video sequences
│   │   ├── video_002/           # Each folder contains frames extracted
│   │   └── ...                  # from the video (e.g. frame_0001.png)
│   └── labels/                  # Corresponding segmentation labels
│       ├── video_001/           # Mirrors the images folder structure
│       ├── video_002/
│       └── ...
├── segmentation_breaking.py     # Dataset loading, preprocessing & frame extraction
└── README.md
```

### Dataset Overview
- **27 video sequences** split across both `images/` and `labels/` folders
- Each video folder contains individual **frames** extracted from the original footage
- Labels are **pixel-wise segmentation masks** corresponding to each image frame
- Total of **13 labelled classes** covering road, vehicles, pedestrians, and more

---

## 📌 Future Improvements

- Improve IoU for underperforming classes (Traffic Signs, Fences, Poles)
- Data augmentation to handle edge cases and rare classes
- Real-time inference optimisation for embedded deployment
- Integration with vehicle CAN bus for live speed control

---

## 📜 License

MIT License — feel free to use and adapt with attribution.

---

## 👤 Author

**Circiu Patrick-Sorin**
