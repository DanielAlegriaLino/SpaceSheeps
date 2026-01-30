from ultralytics import YOLO
import os
from pathlib import Path
from tqdm import tqdm
import cv2

if __name__ == '__main__':
    # Load the trained model
    model = YOLO('custom_model/best.pt')
    
    # Define input directories
    input_dirs = ['train', 'valid']
    
    # Define output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Collect all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    
    for input_dir in input_dirs:
        dir_path = Path(input_dir)
        if dir_path.exists():
            for ext in image_extensions:
                all_images.extend(list(dir_path.glob(f'*{ext}')))
                all_images.extend(list(dir_path.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(all_images)} images to process")
    print(f"Results will be saved to: {output_dir.absolute()}")
    print("="*60)
    
    saved_count = 0
    error_count = 0
    
    # Process each image
    for img_path in tqdm(all_images, desc="Processing images"):
        try:
            # Run inference
            results = model(img_path, conf=0.25)  # confidence threshold
            
            # Get the result object
            result = results[0]
            
            # Create output filename (preserve original name with subfolder prefix)
            parent_folder = img_path.parent.name
            output_filename = f"{parent_folder}_{img_path.name}"
            output_path = output_dir / output_filename
            
            # Plot the results on the image (this includes boxes, even if empty)
            plotted_img = result.plot()
            
            # Save the image with bounding boxes
            cv2.imwrite(str(output_path), plotted_img)
            saved_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            error_count += 1
            continue
    
    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)
    print(f"Processed {len(all_images)} images")
    print(f"Successfully saved: {saved_count} images")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {output_dir.absolute()}")
    print("="*60)

