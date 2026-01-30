from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm

def process_video_to_yolo_format(model, video_path, output_dir, class_names):
    """
    Process a video file and save frames with detections in YOLO format
    
    Args:
        model: YOLO model instance
        video_path: Path to input video file
        output_dir: Directory to save results
        class_names: List of class names
    """
    video_name = Path(video_path).stem
    video_output_dir = output_dir / video_name
    
    # Create subdirectories for images and labels
    images_dir = video_output_dir / 'images'
    labels_dir = video_output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing: {video_path}")
    print(f"Output directory: {video_output_dir}")
    print("="*60)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  - FPS: {fps}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Total frames: {total_frames}")
    print("="*60)
    
    frame_count = 0
    saved_count = 0
    detection_count = 0
    
    # Process frames with progress bar
    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Run inference on the frame
            results = model(frame, conf=0.25, verbose=False)
            result = results[0]
            
            # Frame filename (with leading zeros for proper sorting)
            frame_filename = f"frame_{frame_count:06d}"
            image_path = images_dir / f"{frame_filename}.jpg"
            label_path = labels_dir / f"{frame_filename}.txt"
            
            # Save the original frame without any drawings
            cv2.imwrite(str(image_path), frame)
            
            # Save YOLO format annotations
            with open(label_path, 'w') as f:
                if len(result.boxes) > 0:
                    detection_count += 1
                    
                    for box in result.boxes:
                        # Get box coordinates in xyxy format
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        # Convert to YOLO format (normalized xywh)
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        
                        # Get class id and confidence
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Write YOLO format: class_id x_center y_center width height
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            
            saved_count += 1
            frame_count += 1
            pbar.update(1)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        cap.release()
        pbar.close()
    
    print(f"\nCompleted processing {video_name}:")
    print(f"  - Total frames: {frame_count}")
    print(f"  - Frames with detections: {detection_count}")
    print(f"  - Images saved: {saved_count}")
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
    video_files = ['video1.mp4', 'video2.mp4']
    
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
        print(f"      ├── images/     (annotated frame images)")
        print(f"      └── labels/     (YOLO format .txt files)")
    print("="*60)

