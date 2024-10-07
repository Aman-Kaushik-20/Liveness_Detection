import cv2
from cvzone.FaceDetectionModule import FaceDetector
import os
import shutil

# Configuration
classID = 1  # 0 is fake and 1 is real
inputFolderPath = "Liveness-Dataset\LIVENESS_DATASET\Real"
outputFolderPath = "Filtered_Liveness_data\Filtered_Real"

confidence = 0.8
blurThreshold = 35

offsetPercentageW = 10
offsetPercentageH = 20

floating_points = 6

# Initialize the FaceDetector object
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Create output directory if it doesn't exist
os.makedirs(outputFolderPath, exist_ok=True)

# Initialize counters
total_images = 0
rejected_blurry = 0
rejected_no_face = 0
processed_images = 0

# Process each image in the input directory
for filename in os.listdir(inputFolderPath):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg','.bmp', '.webp')):
        total_images += 1
        # Read the image
        img_path = os.path.join(inputFolderPath, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        # Detect faces in the image
        img, bboxs = detector.findFaces(img, draw=False)

        listBlur = []  # True False values indicating if the faces are blur or not
        listInfo = []  # the normalized values and the class name for the label text file

        # Check if any face is detected
        if bboxs:
            for bbox in bboxs:
                x, y, w, h = bbox['bbox']
                score = bbox["score"][0]

                if score > confidence:
                    # Adding an offset to the face Detected
                    offsetW = (offsetPercentageW / 100) * w
                    x = int(x - offsetW)
                    w = int(w + offsetW * 2)

                    offsetH = (offsetPercentageH / 100) * h
                    y = int(y - offsetH * 3)
                    h = int(h + offsetH * 3.5)

                    # Ensure values are not negative
                    x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)

                    # Find Blurriness
                    imgFace = img[y:y+h, x:x+w]
                    blurvalue = cv2.Laplacian(imgFace, cv2.CV_64F).var()
                    listBlur.append(blurvalue > blurThreshold)

                    # Calculate YOLO format coordinates
                    ih, iw, _ = img.shape
                    x_center = (x + w / 2) / iw
                    y_center = (y + h / 2) / ih
                    width = w / iw
                    height = h / ih

                    # Ensure values are not above 1
                    x_center = min(1, max(0, x_center))
                    y_center = min(1, max(0, y_center))
                    width = min(1, max(0, width))
                    height = min(1, max(0, height))

                    listInfo.append(f"{classID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Save results if faces are detected and not blurry
        if bboxs:
            if listBlur and all(listBlur):  # All faces not blurry
                # Save original image
                output_img_path = os.path.join(outputFolderPath, filename)
                shutil.copy2(img_path, output_img_path)
                
               ## Save Label Text File
                output_txt_path = os.path.join(outputFolderPath, f"{os.path.splitext(filename)[0]}.txt")
                with open(output_txt_path, 'w') as f:
                    f.writelines(listInfo)

                processed_images += 1  # Count processed images
                print(f"Processed and saved: {filename}")
            else:
                rejected_blurry += 1  # Increment blurry images counter
                print(f"Rejected due to blurriness: {filename}")
        else:
            rejected_no_face += 1  # Increment no face detected counter
            print(f"No valid faces detected in: {filename}")

# Print summary of results
print("\nProcessing complete.")
print(f"Total input images: {total_images}")
print(f"Total processed images: {processed_images}")
print(f"Total rejected due to blurriness: {rejected_blurry}")
print(f"Total rejected due to no face detected: {rejected_no_face}")