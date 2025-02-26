import sqlite3
from ultralytics import YOLO
import cv2
import cvzone
import math
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
    "ID", "Name", "Age", "Date of Birth", "Address", "Loyalty",
    "Member Since", "Gender", "Email", "Phone Number", "Membership Type",
    "Status", "Occupation", "Interests", "Marital Status"
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 2
LINE_HEIGHT = 30

cap = cv2.VideoCapture(1)  # Use camera index 1
cap.set(3, 1880)
cap.set(4, 1880)

model = YOLO("best.pt")
classNames = [
    'angela', 'classmate', 'giuliana', 'javier', 'john',
    'maite', 'mike', 'ron', 'shanti', 'tom', 'vilma', 'will'
]

conf_threshold = 0.7
detection_counts = defaultdict(int)
detected_frames = {}

prev_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break

    highest_conf_value = 0
    highest_conf_info = None  
    detections_this_frame = []  

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue  
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            cls = int(box.cls[0])
            profile = getProfile(cls + 1)
            if profile is not None:
                detected_name = profile[1]
                detected_status = profile[11] if len(profile) > 11 else "Inactive"
            else:
                detected_name = classNames[cls] if 0 <= cls < len(classNames) else "Unknown"
                detected_status = "Inactive"
            
            detection_counts[detected_name] += 1

            conf_percent = int(conf * 100)
            label_text = f'{detected_name} - {detected_status} {conf_percent}%'

            # Set bounding box color: Green if Active, Red otherwise
            box_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)

            # 🔴 FIX: Use cv2.rectangle() instead of cvzone.cornerRect()
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)

            # Draw label above bounding box
            cvzone.putTextRect(img, label_text, (max(0, x1), max(35, y1)),
                               scale=2, thickness=2, colorR=box_color)
            
            if conf > highest_conf_value:
                highest_conf_value = conf
                highest_conf_info = {'name': detected_name, 'status': detected_status}
            
            if profile is not None:
                startY = y1 + h + 20
                for i, field_name in enumerate(FIELD_NAMES):
                    if i < len(profile):
                        text = f"{field_name}: {profile[i]}"
                        (text_w, text_h), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
                        cv2.rectangle(img,
                                      (x1, startY + i * LINE_HEIGHT - text_h - 5),
                                      (x1 + text_w, startY + i * LINE_HEIGHT + 5),
                                      (0, 0, 0), cv2.FILLED)
                        cv2.putText(img, text,
                                    (x1, startY + i * LINE_HEIGHT),
                                    FONT, FONT_SCALE, (0, 255, 0), THICKNESS)

            detections_this_frame.append((detected_name, x1, y1, x2, y2))

    if highest_conf_info is not None:
        global_text = f"{highest_conf_info['name']} - {highest_conf_info['status']}"
        global_color = (0, 255, 0) if highest_conf_info['status'].lower() == "active" else (0, 0, 255)
        cv2.putText(img, global_text, (10, 40), FONT, 1.5, global_color, 3)

    for (det_name, x1, y1, x2, y2) in detections_this_frame:
        face_crop = img[y1:y2, x1:x2].copy()
        full_annotated = img.copy()
        detected_frames[det_name] = (full_annotated, face_crop)

    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if detection_counts:
    most_detected = max(detection_counts, key=detection_counts.get)
    if most_detected in detected_frames:
        full_frame, cropped_face = detected_frames[most_detected]
        cv2.imwrite(f"{most_detected}_detected_face.jpg", cropped_face)
        cv2.imwrite(f"{most_detected}_detected_full.jpg", full_frame)

print("Detection counts:", dict(detection_counts))
