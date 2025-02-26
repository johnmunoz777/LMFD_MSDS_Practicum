import sqlite3
import cv2
import cvzone
import time
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class JMModels:
    def __init__(self, db_path="capstone.db",
                 yolo_model_path="best.pt",
                 lbph_recognizer_path="new_recognizer/trainingData.yml"):
        # Set up database and model paths
        self.db_path = db_path
        self.yolo_model_path = yolo_model_path
        self.lbph_recognizer_path = lbph_recognizer_path
        
        # Global constants (from your scripts)
        self.FIELD_NAMES = [
            "ID", "Name", "Age", "Date of Birth", "Address", "Loyalty", "Member Since",
            "Gender", "Email", "Phone Number", "Membership Type", "Status", "Occupation",
            "Interests", "Marital Status"
        ]
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.7
        self.THICKNESS = 2
        self.LINE_HEIGHT = 30
        
        # Load models
        self.yolo_model = YOLO(self.yolo_model_path)
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.lbph_recognizer.read(self.lbph_recognizer_path)
        
        # LBPH confidence threshold (lower is better)
        self.LBPH_CONF_THRESHOLD = 60
        # YOLO confidence threshold (you may override in functions)
        self.YOLO_CONF_THRESHOLD = 0.7

    def getProfile(self, member_id):
        conn = sqlite3.connect(self.db_path)
        cmd = "SELECT * FROM members WHERE id=" + str(member_id)
        cursor = conn.execute(cmd)
        profile = None
        for row in cursor:
            profile = row
        conn.close()
        return profile

    # ----- Function 1: Webcam processing with LBPH (using Haar cascade) -----
    def process_webcam_lbph(self):
        cap = cv2.VideoCapture(1)
        cap.set(3, 1880)
        cap.set(4, 1880)
        detection_counts = defaultdict(int)
        detected_frames = {}
        prev_frame_time = 0

        while True:
            new_frame_time = time.time()
            ret, img = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            best_conf_value = float('inf')
            best_conf_info = None

            for (x, y, w, h) in faces:
                # LBPH recognition
                id_pred, conf = self.lbph_recognizer.predict(gray[y:y+h, x:x+w])
                if conf < self.LBPH_CONF_THRESHOLD:
                    profile = self.getProfile(id_pred)
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

                # Set bounding box color: Green if Active, Red otherwise
                rect_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, 2)
                label_text = f'{detected_name} - {detected_status} {conf_percent}%'
                cvzone.putTextRect(img, label_text, (max(0, x), max(35, y)),
                                   scale=2, thickness=2, colorR=rect_color)

                # Update best detection info (lowest conf is best)
                if conf < best_conf_value:
                    best_conf_value = conf
                    best_conf_info = {'name': detected_name, 'status': detected_status}

                # Draw detailed profile info below the face
                if profile is not None:
                    startY = y + h + 20
                    for i, field_name in enumerate(self.FIELD_NAMES):
                        if i < len(profile):
                            text = f"{field_name}: {profile[i]}"
                            (text_w, text_h), _ = cv2.getTextSize(text, self.FONT, self.FONT_SCALE, self.THICKNESS)
                            cv2.rectangle(img, (x, startY + i * self.LINE_HEIGHT - text_h - 5),
                                          (x + text_w, startY + i * self.LINE_HEIGHT + 5),
                                          (0, 0, 0), cv2.FILLED)
                            cv2.putText(img, text, (x, startY + i * self.LINE_HEIGHT),
                                        self.FONT, self.FONT_SCALE, (0, 255, 0), self.THICKNESS)

                detection_counts[detected_name] += 1
                detected_frames[detected_name] = (img.copy(), img[y:y+h, x:x+w].copy())

            if best_conf_info is not None:
                global_text = f"{best_conf_info['name']} - {best_conf_info['status']}"
                global_color = (0, 255, 0) if best_conf_info['status'].lower() == "active" else (0, 0, 255)
                cv2.putText(img, global_text, (10, 40), self.FONT, 1.5, global_color, 3)

            fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
            prev_frame_time = new_frame_time
            print(f"LBPH Webcam FPS: {fps:.2f}")

            cv2.imshow("Webcam LBPH", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        # Optionally save the best detection's images
        if detection_counts:
            most_detected = max(detection_counts, key=detection_counts.get)
            if most_detected in detected_frames:
                full_frame, cropped_face = detected_frames[most_detected]
                cv2.imwrite(f"{most_detected}_detected_face.jpg", cropped_face)
                cv2.imwrite(f"{most_detected}_detected_full.jpg", full_frame)
        print("LBPH Webcam Detection counts:", dict(detection_counts))

    # ----- Function 2: Webcam processing with YOLO -----
    def process_webcam_yolo(self):
        cap = cv2.VideoCapture(1)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.yolo_model(frame, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < self.YOLO_CONF_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    profile = self.getProfile(int(box.cls[0]) + 1)
                    detected_name = profile[1] if profile else "Unknown"
                    detected_status = profile[11] if profile and len(profile) > 11 else "Inactive"
                    color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cvzone.putTextRect(frame, f"{detected_name} - {detected_status}", (x1, y1 - 10),
                                       scale=1, thickness=2, colorR=color)
            cv2.imshow("Webcam YOLO", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # ----- Function 3: Process video file using YOLO -----
    def process_video_yolo(self, input_video_path, output_video_path,
                           rotate_mode=0, label_output_model="YOLO Model Output",
                           adjust=1.0, resize_video=None, conf_threshold=None):
        # Use default YOLO confidence threshold if not provided
        if conf_threshold is None:
            conf_threshold = self.YOLO_CONF_THRESHOLD

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

        detection_counts = defaultdict(int)
        detected_frames = {}
        prev_frame_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate/flip frame if requested
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

            results = self.yolo_model(frame, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    profile = self.getProfile(int(box.cls[0]) + 1)
                    if profile is not None:
                        detected_name = profile[1]
                        detected_status = profile[11] if len(profile) > 11 else "Inactive"
                    else:
                        detected_name = "Unknown"
                        detected_status = "Inactive"
                    detection_counts[detected_name] += 1
                    conf_percent = int(conf * 100)
                    label_text = f"{detected_name} - {detected_status} {conf_percent}%"
                    box_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                    cvzone.putTextRect(frame, label_text, (max(0, x1), max(35, y1)),
                                       scale=2, thickness=2, colorR=box_color)
                    if conf > highest_conf_value:
                        highest_conf_value = conf
                        highest_conf_info = {'name': detected_name, 'status': detected_status}
                    if profile is not None:
                        startY = y1 + (y2 - y1) + 20
                        for i, field_name in enumerate(self.FIELD_NAMES):
                            if i < len(profile):
                                text = f"{field_name}: {profile[i]}"
                                (text_w, text_h), _ = cv2.getTextSize(text, self.FONT, self.FONT_SCALE, self.THICKNESS)
                                cv2.rectangle(frame,
                                              (x1, startY + i * self.LINE_HEIGHT - text_h - 5),
                                              (x1 + text_w, startY + i * self.LINE_HEIGHT + 5),
                                              (0, 0, 0), cv2.FILLED)
                                cv2.putText(frame, text,
                                            (x1, startY + i * self.LINE_HEIGHT),
                                            self.FONT, self.FONT_SCALE, (0, 255, 0), self.THICKNESS)
                    detections_this_frame.append((detected_name, x1, y1, x2, y2))

            if highest_conf_info is not None:
                global_text = f"{highest_conf_info['name']} - {highest_conf_info['status']}"
                global_color = (0, 255, 0) if highest_conf_info['status'].lower() == "active" else (0, 0, 255)
            else:
                global_text = label_output_model
                global_color = (255, 255, 255)
            cv2.putText(frame, global_text, (10, 40), self.FONT, 1.5, global_color, 3)

            for (det_name, x1, y1, x2, y2) in detections_this_frame:
                face_crop = frame[y1:y2, x1:x2].copy()
                detected_frames[det_name] = (frame.copy(), face_crop)

            if adjust != 1.0:
                display_frame = cv2.resize(frame, (0, 0), fx=adjust, fy=adjust)
            else:
                display_frame = frame

            cv2.imshow("Processed Video - YOLO", display_frame)
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
        print("YOLO Video Detection counts:", dict(detection_counts))

    # ----- Function 4: Process video file using LBPH (cascade + recognizer) -----
    def process_video_lbph(self, input_video_path, output_video_path,
                           rotate_mode=0, label_output_model="Face Recognition Model Output",
                           adjust=1.0, resize_video=None, conf_threshold=50):
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

        facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Use a fresh recognizer instance here (or reuse self.lbph_recognizer)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.lbph_recognizer_path)

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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            highest_conf_value = float("inf")
            highest_conf_info = None
            detections_this_frame = []

            for (x, y, w, h) in faces:
                id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                if conf < conf_threshold:
                    profile = self.getProfile(id)
                    detected_name = profile[1] if profile else f"ID {id}"
                    detected_status = profile[11] if profile and len(profile) > 11 else "Inactive"
                else:
                    detected_name = "Unknown"
                    detected_status = "Inactive"
                detection_counts[detected_name] += 1
                conf_percent = int(100 - conf)
                label_text = f"{detected_name} - {detected_status} {conf_percent}%"
                rect_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
                cvzone.putTextRect(frame, label_text, (max(0, x), max(35, y)),
                                   scale=2, thickness=2, colorR=rect_color)
                if conf < highest_conf_value:
                    highest_conf_value = conf
                    highest_conf_info = {'name': detected_name, 'status': detected_status}
                if profile is not None:
                    startY = y + h + 20
                    for i, field_name in enumerate(self.FIELD_NAMES):
                        if i < len(profile):
                            text = f"{field_name}: {profile[i]}"
                            (text_w, text_h), _ = cv2.getTextSize(text, self.FONT, self.FONT_SCALE, self.THICKNESS)
                            cv2.rectangle(frame,
                                          (x, startY + i * self.LINE_HEIGHT - text_h - 5),
                                          (x + text_w, startY + i * self.LINE_HEIGHT + 5),
                                          (0, 0, 0), cv2.FILLED)
                            cv2.putText(frame, text,
                                        (x, startY + i * self.LINE_HEIGHT),
                                        self.FONT, self.FONT_SCALE, (0, 255, 0), self.THICKNESS)
                detections_this_frame.append((detected_name, x, y, x+w, y+h))

            if highest_conf_info is not None:
                global_text = f"{highest_conf_info['name']} - {highest_conf_info['status']}"
                global_color = (0, 255, 0) if highest_conf_info['status'].lower() == "active" else (0, 0, 255)
            else:
                global_text = label_output_model
                global_color = (255, 255, 255)
            cv2.putText(frame, global_text, (10, 40), self.FONT, 1.5, global_color, 3)

            for (det_name, x1, y1, x2, y2) in detections_this_frame:
                face_crop = frame[y1:y2, x1:x2].copy()
                detected_frames[det_name] = (frame.copy(), face_crop)

            if adjust != 1.0:
                display_frame = cv2.resize(frame, (0, 0), fx=adjust, fy=adjust)
            else:
                display_frame = frame

            cv2.imshow("Processed Video - LBPH", display_frame)
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
        print("LBPH Video Detection counts:", dict(detection_counts))

    # ----- Function 5: Play a video without detection -----
    def play_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Playing Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# ===== Example Usage =====
if __name__ == "__main__":
    # Create an instance of the JMModels class
    jm = JMModels()
    
 
    
    # 1. Process Webcam with LBPH
    # jm.process_webcam_lbph()
    
    # 2. Process Webcam with YOLO
    # jm.process_webcam_yolo()
    
    # 3. Process video file with YOLO
    # jm.process_video_yolo("C:/Users/johnm/capstone/volunteers/ron/ron.mp4", 
    #                       "output_video_ron_yolo.mp4", rotate_mode=1, 
    #                       label_output_model="YOLO Model Output", adjust=0.8, 
    #                       resize_video=(1280,1280), conf_threshold=0.7)
    
    # 4. Process video file with LBPH
    # jm.process_video_lbph("C:/Users/johnm/capstone/volunteers/ron/ron.mp4", 
    #                       "output_video_ron_lbph.mp4", rotate_mode=1, 
    #                       label_output_model="Face Recognition Model Output", adjust=0.8, 
    #                       resize_video=(1280,1280), conf_threshold=50)
    
    # 5. Play a video (no detection)
    # jm.play_video("sample.mp4")
