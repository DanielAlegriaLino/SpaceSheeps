from ultralytics import YOLO
import cv2
from pathlib import Path

if __name__ == '__main__':
    # Load the trained model
    print("Loading model...")
    model = YOLO('custom_model/best.pt')
    
    # Input video path
    video_path = 'video2.mp4'
    
    # Output video path
    output_path = 'video2_detected.mp4'
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
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
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_count = 0
    
    print("Processing video frames...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Run inference on the frame
            results = model(frame, conf=0.25, verbose=False)
            
            # Get the result object
            result = results[0]
            
            # Count detections in this frame
            if len(result.boxes) > 0:
                detection_count += 1
            
            # Plot the results on the frame
            plotted_frame = result.plot()
            
            # Write the frame with detections to output video
            out.write(plotted_frame)
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        # Release everything
        cap.release()
        out.release()
    
    print("\n" + "="*60)
    print("Video Processing Complete!")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with detections: {detection_count}")
    print(f"Output saved to: {output_path}")
    print("="*60)

