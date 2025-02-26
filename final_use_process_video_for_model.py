import os
import cv2

def get_last_sample_number(user_id, video_data_folder):
    existing_files = [f for f in os.listdir(video_data_folder) if f.startswith(f"User.{user_id}.") and f.endswith(".jpg")]
    if not existing_files:
        return 0
    sample_numbers = [int(f.split('.')[2]) for f in existing_files]
    return max(sample_numbers)

def detect_faces_and_save(user_id, video_data_folder, num_images, display_size=(640, 480)):
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    sampleNum = get_last_sample_number(user_id, video_data_folder)  
    print(f"Starting from sample number: {sampleNum + 1}")
    cam = cv2.VideoCapture(1)
    captured_images = 0
    while captured_images < num_images:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            sampleNum += 1
            face_region = gray[y:y+h, x:x+w]
            new_img_name = f"User.{user_id}.{sampleNum}.jpg"
            cv2.imwrite(os.path.join(video_data_folder, new_img_name), face_region)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            captured_images += 1
            print(f"Captured {captured_images}/{num_images} images.")
        resized_display = cv2.resize(frame, display_size)
        cv2.imshow("Face Detection - Webcam", resized_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def capture_faces_from_video(video_path, user_id, num_images=20, display_size=(640, 480)):
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    sampleNum = get_last_sample_number(user_id, "video_dataset")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames in Video: {total_frames}")
    captured_images = 0
    while cap.isOpened() and captured_images < num_images:
        ret, img = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(f"video_dataset/User.{user_id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            captured_images += 1
            print(f"Captured {captured_images}/{num_images} images.")
        resized_display = cv2.resize(img, display_size)
        cv2.imshow("Face", resized_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def capture_faces_from_rotated_video(video_path, user_id, num_images=20, display_size=(640, 480)):
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    sampleNum = get_last_sample_number(user_id, "video_dataset")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames in Video: {total_frames}")
    captured_images = 0
    while cap.isOpened() and captured_images < num_images:
        ret, img = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            sampleNum += 1
            face_region = gray[y:y+h, x:x+w]
            cv2.imwrite(f"video_dataset/User.{user_id}.{sampleNum}.jpg", face_region)
            cv2.rectangle(img_rotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            captured_images += 1
            print(f"Captured {captured_images}/{num_images} images.")
        resized_display = cv2.resize(img_rotated, display_size)
        cv2.imshow("Face Detection", resized_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def preview_video_with_delay(video_path, delay=30, display_size=(640, 480)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame or end of video")
            break
        resized_display = cv2.resize(frame, display_size)
        cv2.imshow("Video Preview", resized_display)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    
    
############ Still Images
def detect_still_images(user_id, images_folder, video_data_folder, display_size=(1280, 720)):
    """
    Detect faces in images, rename them according to the video_data folder format, 
    and save them in the video_data folder.
    """
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Get the last sample number used for the user
    sampleNum = get_last_sample_number(user_id, video_data_folder)  
    print(f"Starting from sample number: {sampleNum + 1}")

    # List all image files in the still_images folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
    print(f"Found {len(image_files)} images to process.")

    # Loop through each image
    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to read image {img_file}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        print(f"Faces detected in {img_file}: {len(faces)}")  # Debugging step

        # Process each face found in the image
        for (x, y, w, h) in faces:
            sampleNum += 1  # Increment sample number for each detected face
            face_region = gray[y:y+h, x:x+w]
            
            # Save the detected face with a unique name
            new_img_name = f"User.{user_id}.{sampleNum}.jpg"
            cv2.imwrite(os.path.join(video_data_folder, new_img_name), face_region)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with the face rectangles
        resized_display = cv2.resize(img, display_size)  # Only resize for display
        cv2.imshow("Face Detection", resized_display)
        cv2.waitKey(1)

    cv2.destroyAllWindows()





#4863
my_f="bet.mp4"
user_id = 5
#num_images = 10
video_data_folder = "video_dataset"
images_folder = "C:/Users/johnm/capstone/volunteers/john_still_images"  # Your folder where images are located
#preview_video_with_delay(my_f, delay=30, display_size=(1280, 1280))
#capture_faces_from_rotated_video(my_f, user_id, num_images=20, display_size=(1280, 1280))
detect_faces_and_save(user_id, video_data_folder, num_images=200, display_size=(1280, 1280))
#detect_still_images(user_id, images_folder, video_data_folder, display_size=(1280, 1280))