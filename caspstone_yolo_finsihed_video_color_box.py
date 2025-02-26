import sqlite3
from ultralytics import YOLO
import cv2
import cvzone
import time
import os
from collections import defaultdict

# Global constants
FIELD_NAMES = [
    "ID", "Name", "Age", "Date of Birth", "Address", "Loyalty",
    "Member Since", "Gender", "Email", "Phone Number", "Membership Type",
    "Status", "Occupation", "Interests", "Marital Status"
]
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 2
LINE_HEIGHT = 30

def getProfile(member_id):
    conn = sqlite3.connect('capstone.db')
    cmd = "SELECT * FROM members WHERE id=" + str(member_id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

def process_video(input_video_path, output_video_path,
                  rotate_mode=0,
                  label_output_yolo_model="",
                  adjust=1.0,
                  resize_video=None,
                  conf_threshold=0.3):
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Original video shape: {frame_width} x {frame_height}")

    if resize_video is not None:
        out_width, out_height = resize_video
    else:
        out_width, out_height = frame_width, frame_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    model = YOLO("best.pt")
    classNames = [
        'angela', 'classmate', 'giuliana', 'javier', 'john',
        'maite', 'mike', 'ron', 'shanti', 'tom', 'vilma', 'will'
    ]

    detection_counts = defaultdict(int)
    detected_frames = {}  
    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rotate_mode == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_mode == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotate_mode == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate_mode == 4:
            frame = cv2.flip(frame, 1)

        new_frame_time = time.time()

        highest_conf_value = 0
        highest_conf_info = None  

        detections_this_frame = []

        results = model(frame, stream=True)
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
                label_text = f"{detected_name} - {detected_status} {conf_percent}%"
                text_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)

                # ✅ FIX: Draw bounding box manually (Green if Active, Red otherwise)
                box_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)  

                cvzone.putTextRect(frame, label_text, (max(0, x1), max(35, y1)),
                                   scale=2, thickness=2, colorR=text_color)

                if conf > highest_conf_value:
                    highest_conf_value = conf
                    highest_conf_info = {'name': detected_name, 'status': detected_status}

                if profile is not None:
                    startY = y1 + h + 20
                    for i, field_name in enumerate(FIELD_NAMES):
                        if i < len(profile):
                            text = f"{field_name}: {profile[i]}"
                            (text_w, text_h), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
                            cv2.rectangle(frame,
                                          (x1, startY + i * LINE_HEIGHT - text_h - 5),
                                          (x1 + text_w, startY + i * LINE_HEIGHT + 5),
                                          (0, 0, 0), cv2.FILLED)
                            cv2.putText(frame, text,
                                        (x1, startY + i * LINE_HEIGHT),
                                        FONT, FONT_SCALE, (0, 255, 0), THICKNESS)

                detections_this_frame.append((detected_name, x1, y1, x2, y2))

        if highest_conf_info is not None:
            global_text = f"{highest_conf_info['name']} - {highest_conf_info['status']}"
            global_color = (0, 255, 0) if highest_conf_info['status'].lower() == "active" else (0, 0, 255)
        else:
            global_text = label_output_yolo_model if label_output_yolo_model else ""
            global_color = (255, 255, 255)
        cv2.putText(frame, global_text, (10, 40), FONT, 1.5, global_color, 3)

        for (det_name, x1, y1, x2, y2) in detections_this_frame:
            face_crop = frame[y1:y2, x1:x2].copy()
            detected_frames[det_name] = (frame.copy(), face_crop)

        if adjust != 1.0:
            display_frame = cv2.resize(frame, (0, 0), fx=adjust, fy=adjust)
        else:
            display_frame = frame

        cv2.imshow("Processed Video", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if resize_video is not None:
            output_frame = cv2.resize(frame, (out_width, out_height))
        else:
            output_frame = frame
        out.write(output_frame)

        prev_frame_time = new_frame_time

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if detection_counts:
        most_detected = max(detection_counts, key=detection_counts.get)
        if most_detected in detected_frames:
            full_frame, cropped_face = detected_frames[most_detected]
            cv2.imwrite(f"{most_detected}_detected_face.jpg", cropped_face)
            cv2.imwrite(f"{most_detected}_detected_full.jpg", full_frame)

    print("Detection counts:", dict(detection_counts))

iv="C:/Users/johnm/capstone/volunteers/ron/ron.mp4"

process_video(iv, "output_video_ron.mp4", rotate_mode=1, 
              label_output_yolo_model="YOLO Model Output", 
              adjust=0.8, resize_video=(1280,1280), conf_threshold=0.7)
