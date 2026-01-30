# YOLO Space Debris Detection Training

This project trains a YOLOv8 model to detect space debris, satellites, and asteroids.

## Dataset

- **Classes**: 3 (space_debris, statelites, asteroids)
- **Training Images**: ~498 images with annotations
- **Format**: YOLO format (normalized bounding boxes)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Dataset Structure

Your dataset should have the following structure:
```
yolodefspace/
├── train/
│   ├── image1.jpg
│   ├── image1.txt  (YOLO format labels)
│   ├── image2.jpg
│   ├── image2.txt
│   └── ...
├── data.yaml
├── train.py
└── requirements.txt
```

## Training

### Basic Training

Run the training script:

```bash
python train.py
```

### Customization

You can modify the following parameters in `train.py`:

- **Model Size**: Change `yolov8n.pt` to `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, or `yolov8x.pt`
  - `n` (nano): Fastest, smallest
  - `s` (small): Good balance
  - `m` (medium): Better accuracy
  - `l` (large): Higher accuracy
  - `x` (xlarge): Best accuracy, slowest

- **Epochs**: Adjust `epochs=100` (default: 100)
- **Batch Size**: Adjust `batch=16` based on GPU memory
- **Image Size**: Adjust `imgsz=640` (640, 1280, etc.)
- **Device**: Set `device=0` for GPU or `device='cpu'` for CPU

### Training on CPU

If you don't have a GPU, change in `train.py`:
```python
device='cpu',  # instead of device=0
batch=4,       # reduce batch size for CPU
```

## Results

After training, results will be saved to:
- `runs/train/yolo_space_debris/weights/best.pt` - Best model weights
- `runs/train/yolo_space_debris/weights/last.pt` - Last epoch weights
- `runs/train/yolo_space_debris/` - Training metrics, confusion matrix, etc.

## Inference

After training, you can use the model for predictions:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/train/yolo_space_debris/weights/best.pt')

# Run inference
results = model('path/to/test/image.jpg')

# Display results
results[0].show()

# Save results
results[0].save('output.jpg')
```

## Notes

- The training uses the same images for validation since no separate validation set is provided
- Consider splitting your data into train/val sets for better evaluation
- Training time depends on your hardware (GPU recommended)
- The model will automatically download pretrained YOLOv8 weights on first run

## Classes

0. space_debris
1. statelites (satellites)
2. asteroids

