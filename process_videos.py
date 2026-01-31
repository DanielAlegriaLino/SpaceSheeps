from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm

BOX_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


def process_video_to_yolo_format(model, video_path, output_dir, class_names):
    """
    Process a video file: run YOLO detection and output an annotated video.
    """
    video_name = Path(video_path).stem
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = video_output_dir / f"{video_name}_detected.mp4"

    print(f"\nProcessing: {video_path}")
    print(f"Output video: {output_video_path}")
    print("="*60)

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  - FPS: {fps}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Total frames: {total_frames}")
    print("="*60)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_count = 0
    detection_count = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame, conf=0.25, verbose=False)
            result = results[0]

            if len(result.boxes) > 0:
                detection_count += 1

                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    color = BOX_COLORS[class_id % len(BOX_COLORS)]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{class_names[class_id]} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            out.write(frame)
            frame_count += 1
            pbar.update(1)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")

    finally:
        cap.release()
        out.release()
        pbar.close()

    print(f"\nCompleted processing {video_name}:")
    print(f"  - Total frames: {frame_count}")
    print(f"  - Frames with detections: {detection_count}")
    print(f"  - Output video: {output_video_path}")
    print("="*60)

    return frame_count, detection_count


if __name__ == '__main__':
    # Load the trained model
    print("Loading model...")
    model = YOLO('custom_model/best.pt')
    print("Model loaded successfully!")
    print("="*60)
    
    # Get class names
    class_names = model.names
    print(f"Classes: {class_names}")
    print("="*60)
    
    # Define output directory
    output_dir = Path('Results')
    output_dir.mkdir(exist_ok=True)
    
    # Video files to process
    video_files = ['chino_video.mp4']
    
    # Check which videos exist
    existing_videos = [v for v in video_files if Path(v).exists()]
    
    if not existing_videos:
        print("ERROR: No video files found!")
        exit(1)
    
    print(f"Found {len(existing_videos)} video(s) to process: {existing_videos}")
    print("="*60)
    
    # Process each video
    total_frames = 0
    total_detections = 0
    
    for video_path in existing_videos:
        frames, detections = process_video_to_yolo_format(
            model, 
            video_path, 
            output_dir, 
            class_names
        )
        total_frames += frames
        total_detections += detections
    
    # Final summary
    print("\n" + "="*60)
    print("ALL VIDEOS PROCESSED!")
    print("="*60)
    print(f"Total frames processed: {total_frames}")
    print(f"Total frames with detections: {total_detections}")
    print(f"Detection rate: {(total_detections/total_frames*100):.2f}%" if total_frames > 0 else "N/A")
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("="*60)
    print("\nFolder structure:")
    print("Results/")
    for video_file in existing_videos:
        video_name = Path(video_file).stem
        print(f"  └── {video_name}/")
        print(f"      └── {video_name}_detected.mp4")
    print("="*60)

