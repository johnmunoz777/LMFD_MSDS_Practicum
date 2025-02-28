import os
import cv2
import numpy as np
from PIL import Image

# Initialize Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "C:/Users/johnm/capstone/video_dataset"

def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faces = []
    IDs = []

    print(f"Found {len(imagePaths)} images to process.")
    
    for i, imagePath in enumerate(imagePaths):
        try:
            print(f"Processing ({i+1}/{len(imagePaths)}): {imagePath}")

            # Load and convert image
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')

            # Extract ID from filename
            filename = os.path.basename(imagePath)
            parts = filename.split('.')
            
            if len(parts) < 3:
                print(f"⚠️ Skipping {filename} - Incorrect filename format")
                continue

            ID = int(parts[1])  

            # Validate image array
            if faceNp is None or faceNp.size == 0:
                print(f"⚠️ Skipping {filename} - Image could not be loaded properly")
                continue

            # Append data
            faces.append(faceNp)
            IDs.append(ID)

            # Show image
            cv2.imshow("Training", faceNp)
            
            # **Handle closing properly**
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                print("🛑 Training stopped by user.")
                cv2.destroyAllWindows()
                return np.array(IDs), faces  # Stop processing immediately
        
        except Exception as e:
            print(f"❌ Error processing {imagePath}: {e}")
            continue  

    # **Close the window after the last image**
    cv2.destroyAllWindows()
    return np.array(IDs), faces

# Get Images and IDs
IDS, FACES = getImagesWithID(path)

# Train if data is available
if len(FACES) > 0:
    recognizer.train(FACES, IDS)
    recognizer.save('recognizer/trainingdata.yml')
    print("✅ Training complete. Data saved.")
else:
    print("❌ No valid images found. Training skipped.")

cv2.destroyAllWindows()  # Force close after loop
