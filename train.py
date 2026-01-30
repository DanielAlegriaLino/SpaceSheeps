from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # You can use yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), 
    # yolov8l.pt (large), or yolov8x.pt (xlarge)
    model = YOLO('yolov8n.pt')  # Starting with nano model for faster training

    # Train the model
    results = model.train(
    data='data.yaml',           # path to dataset config file
    epochs=10,                  # number of training epochs
    imgsz=640,                   # image size
    batch=16,                     # batch size (reduced for CPU training)
    name='yolo_space_debris',    # experiment name
    patience=5,                 # early stopping patience
    save=True,                   # save checkpoints
    device=0,                    # GPU device (0 for first GPU, 'cpu' for CPU)
    project='runs/train',        # project directory
    exist_ok=True,               # overwrite existing experiment
    pretrained=True,             # use pretrained weights
    optimizer='auto',            # optimizer (auto, SGD, Adam, AdamW)
    verbose=True,                # verbose output
    seed=42,                     # random seed for reproducibility
    deterministic=True,          # deterministic training
    single_cls=False,            # train as single-class dataset
    rect=False,                  # rectangular training
    cos_lr=False,                # cosine learning rate scheduler
    close_mosaic=10,             # disable mosaic augmentation for final N epochs
    resume=False,                # resume training from last checkpoint
    amp=True,                    # automatic mixed precision training
    fraction=1.0,                # train on fraction of data
    profile=False,               # profile ONNX and TensorRT speeds
    freeze=None,                 # freeze layers (list of layer indices)
    # Learning rate settings
    lr0=0.01,                    # initial learning rate
    lrf=0.01,                    # final learning rate (lr0 * lrf)
    momentum=0.937,              # SGD momentum/Adam beta1
    weight_decay=0.0005,         # optimizer weight decay
    warmup_epochs=3.0,           # warmup epochs
    warmup_momentum=0.8,         # warmup initial momentum
    warmup_bias_lr=0.1,          # warmup initial bias lr
    # Augmentation settings
    hsv_h=0.015,                 # HSV-Hue augmentation
    hsv_s=0.7,                   # HSV-Saturation augmentation
    hsv_v=0.4,                   # HSV-Value augmentation
    degrees=0.0,                 # rotation (+/- deg)
    translate=0.1,               # translation (+/- fraction)
    scale=0.5,                   # scale (+/- gain)
    shear=0.0,                   # shear (+/- deg)
    perspective=0.0,             # perspective (+/- fraction)
    flipud=0.0,                  # vertical flip (probability)
    fliplr=0.5,                  # horizontal flip (probability)
    mosaic=1.0,                  # mosaic augmentation (probability)
    mixup=0.0,                   # mixup augmentation (probability)
    copy_paste=0.0,              # copy-paste augmentation (probability)
    )

    # Print results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Results saved to: {results.save_dir}")
    print(f"Best model weights: {results.save_dir}/weights/best.pt")
    print(f"Last model weights: {results.save_dir}/weights/last.pt")
    print("="*60)

