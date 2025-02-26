import sqlite3
import cv2
import cvzone
import time
import os
from collections import defaultdict

print(os.getcwd())

def getProfile(member_id):
    conn = sqlite3.connect('capstone.db')
    cmd = "SELECT * FROM members WHERE id=" + str(member_id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

FIELD_NAMES = [
    "ID", "Name", "Age", "Date of Birth", "Address", "Loyalty", "Member Since",
    "Gender", "Email", "Phone Number", "Membership Type", "Status", "Occupation",
    "Interests", "Marital Status"
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 2
LINE_HEIGHT = 30

# Open webcam (camera index 1)
cap = cv2.VideoCapture(1)
cap.set(3, 1880)
cap.set(4, 1880)

# Load the Haar cascade for face detection and LBPH recognizer
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('new_recognizer/trainingData.yml')

# Confidence threshold for LBPH (lower = better match)
CONFIDENCE_THRESHOLD = 60

# Tracking detections and storing images
detection_counts = defaultdict(int)
detected_frames = {}

prev_frame_time = 0

while True:
    new_frame_time = time.time()
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    best_conf_value = float('inf')
    best_conf_info = None

    for (x, y, w, h) in faces:
        # LBPH recognition
        id_pred, conf = recognizer.predict(gray[y:y+h, x:x+w])

        if conf < CONFIDENCE_THRESHOLD:
            profile = getProfile(id_pred)
            if profile is not None:
                detected_name = profile[1]
                detected_status = profile[11] if len(profile) > 11 else "Inactive"
            else:
                detected_name = str(id_pred)
                detected_status = "Inactive"
            conf_percent = max(0, min(100, 100 - int(conf)))
        else:
            profile = None
            detected_name = "Unknown"
            detected_status = "Inactive"
            conf_percent = 0

        # ✅ Set bounding box color: Green if Active, Red otherwise
        rect_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)

        # Draw the bounding box with the determined color
        cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, 2)

        # Prepare the label text and draw it.
        label_text = f'{detected_name} - {detected_status} {conf_percent}%'
        cvzone.putTextRect(img, label_text, (max(0, x), max(35, y)),
                           scale=2, thickness=2, colorR=rect_color)

        # Update best detection info (lowest conf is best for LBPH).
        if conf < best_conf_value:
            best_conf_value = conf
            best_conf_info = {'name': detected_name, 'status': detected_status}

        # Draw detailed profile info below the face rectangle if profile is available.
        if profile is not None:
            startY = y + h + 20
            for i, field_name in enumerate(FIELD_NAMES):
                if i < len(profile):
                    text = f"{field_name}: {profile[i]}"
                    (text_w, text_h), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
                    cv2.rectangle(img, (x, startY + i * LINE_HEIGHT - text_h - 5),
                                  (x + text_w, startY + i * LINE_HEIGHT + 5), (0, 0, 0), cv2.FILLED)
                    cv2.putText(img, text, (x, startY + i * LINE_HEIGHT),
                                FONT, FONT_SCALE, (0, 255, 0), THICKNESS)

        # Store the detected frame and face crop
        detection_counts[detected_name] += 1
        detected_frames[detected_name] = (img.copy(), img[y:y+h, x:x+w].copy())

    # Draw the global overlay at the top left using the best detection info.
    if best_conf_info is not None:
        global_text = f"{best_conf_info['name']} - {best_conf_info['status']}"
        global_color = (0, 255, 0) if best_conf_info['status'].lower() == "active" else (0, 0, 255)
        cv2.putText(img, global_text, (10, 40), FONT, 1.5, global_color, 3)

    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save images for the most detected person.
if detection_counts:
    most_detected = max(detection_counts, key=detection_counts.get)
    if most_detected in detected_frames:
        full_frame, cropped_face = detected_frames[most_detected]
        if cropped_face is not None and cropped_face.size != 0:
            cv2.imwrite(f"{most_detected}_detected_face.jpg", cropped_face)
        else:
            print(f"No valid cropped face for {most_detected}; skipping face image save.")
        if full_frame is not None and full_frame.size != 0:
            cv2.imwrite(f"{most_detected}_detected_full.jpg", full_frame)
        else:
            print(f"Full annotated frame for {most_detected} is empty.")

print("Detection counts:", dict(detection_counts))
