import os
import json
import cv2
import joblib
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from utils.landmark_utils import extract_features_from_landmarks, extract_landmarks_from_results, validate_feature_vector

class SignLanguageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition System")
        self.root.geometry("1000x620")
        self.root.resizable(False, False)

        # ---------------------------------------------------------
        # 1. Initialization & Model Loading
        # ---------------------------------------------------------
        self.model_path = os.path.join("models", "sign_classifier.joblib")
        self.labels_path = os.path.join("models", "labels.json")

        self.model = self.load_model()
        self.labels = self.load_labels()

        # MediaPipe Hands Setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Pipeline Variables
        self.cap = None
        self.is_running = False
        self.history = []
        self.window_size = 10  # Window size for calculating stable prediction

        # ---------------------------------------------------------
        # 2. GUI Layout
        # ---------------------------------------------------------
        # Main Container Frames
        self.camera_frame = tk.Frame(self.root, width=640, height=480, bg="black")
        self.camera_frame.pack(side=tk.LEFT, padx=15, pady=15)
        self.camera_frame.pack_propagate(False)

        self.sidebar_frame = tk.Frame(self.root, width=300)
        self.sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=15, pady=15)

        # Video Screen Label
        self.video_display = tk.Label(
            self.camera_frame, 
            text="Camera Offline\nClick 'Start Camera' to begin", 
            bg="black", 
            fg="white", 
            font=("Arial", 14)
        )
        self.video_display.pack(expand=True, fill=tk.BOTH)

        # --- Sidebar: Metrics Section ---
        tk.Label(self.sidebar_frame, text="Live Output", font=("Arial", 16, "bold")).pack(anchor="w", pady=(0, 15))

        self.val_current = self.create_metric_row("Current prediction:", "N/A")
        self.val_stable = self.create_metric_row("Stable prediction:", "N/A")
        self.val_confidence = self.create_metric_row("Confidence:", "0.00")
        self.val_hand = self.create_metric_row("Hand detected:", "NO")
        self.val_status = self.create_metric_row("Status:", "Stopped")

        # Separator
        ttk.Separator(self.sidebar_frame, orient="horizontal").pack(fill=tk.X, pady=20)

        # --- Sidebar: Control Buttons ---
        tk.Label(self.sidebar_frame, text="Controls", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))

        self.btn_start = ttk.Button(self.sidebar_frame, text="Start Camera", command=self.start_camera)
        self.btn_start.pack(fill=tk.X, pady=5)

        self.btn_stop = ttk.Button(self.sidebar_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=5)

        self.btn_reset = ttk.Button(self.sidebar_frame, text="Reset Prediction", command=self.reset_prediction)
        self.btn_reset.pack(fill=tk.X, pady=5)

        self.btn_quit = ttk.Button(self.sidebar_frame, text="Quit", command=self.quit_app)
        self.btn_quit.pack(fill=tk.X, pady=5)

        # Intercept window close button
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    # ---------------------------------------------------------
    # Helper & UI Creation Methods
    # ---------------------------------------------------------
    
    def load_labels(self):
        """Loads label names from models/labels.json"""
        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                messagebox.showerror("Labels Load Error", f"Failed to load labels:\n{e}")
                return []
        else:
            messagebox.showwarning(
                "Labels Warning",
                f"Labels file not found at: {self.labels_path}"
            )
            return []
    
    def load_model(self):
        """Loads model from models/sign_classifier.joblib"""
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except Exception as e:
                messagebox.showerror("Model Load Error", f"Failed to load model:\n{e}")
                return None
        else:
            messagebox.showwarning("Model Warning", f"Model file not found at: {self.model_path}")
            return None

    def create_metric_row(self, title, default_value):
        row = tk.Frame(self.sidebar_frame)
        row.pack(fill=tk.X, pady=4)
        
        lbl_title = tk.Label(row, text=title, font=("Arial", 11, "bold"))
        lbl_title.pack(side=tk.LEFT)
        
        lbl_value = tk.Label(row, text=default_value, font=("Arial", 11), fg="#0066CC")
        lbl_value.pack(side=tk.RIGHT)
        
        return lbl_value

    # ---------------------------------------------------------
    # Real-Time Processing Loop
    # ---------------------------------------------------------
    def process_video_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            # 1. Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # 2. Extract MediaPipe landmarks
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            current_pred = "N/A"
            confidence = 0.0
            hand_detected = False

            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw skeletal overlay on frame
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract raw landmark coordinates (21 joints * 3 axes)
                    landmark_arrays = extract_landmarks_from_results(results)
                    if landmark_arrays:
                        features = extract_features_from_landmarks(landmark_arrays[0])

                    # 3. Model Inference
                    if features is not None and validate_feature_vector(features):
                        if self.model is not None:
                            try:
                                feature_row = np.asarray(features, dtype=np.float32).reshape(1, -1)

                                predicted_id = int(self.model.predict(feature_row)[0])

                                if 0 <= predicted_id < len(self.labels):
                                    current_pred = self.labels[predicted_id]

                                if hasattr(self.model, "predict_proba"):
                                    probabilities = self.model.predict_proba(feature_row)[0]
                                    class_ids = [
                                        int(class_id) for class_id in self.model.classes_
                                    ]

                                    probability_index = class_ids.index(predicted_id)
                                    confidence = float(probabilities[probability_index])

                                self.history.append(current_pred)

                                if len(self.history) > self.window_size:
                                    self.history.pop(0)

                            except:
                                if not hand_detected:
                                    self.history.clear()

            # Calculate stable prediction via majority vote
            stable_pred = "N/A"
            if self.history:
                stable_pred = max(set(self.history), key=self.history.count)

            # 4. Update UI Displays
            self.val_current.config(text=current_pred)
            self.val_stable.config(text=stable_pred)
            self.val_confidence.config(text=f"{confidence:.2f}")
            self.val_hand.config(text="YES" if hand_detected else "NO")

            # 5. Render OpenCV frame to Tkinter widget
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.video_display.img_tk = img_tk
            self.video_display.configure(image=img_tk)

        # Loop frame update (~30 FPS target)
        self.root.after(30, self.process_video_frame)

    # ---------------------------------------------------------
    # Button Callbacks
    # ---------------------------------------------------------
    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not access the webcam.")
                return

            self.is_running = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.val_status.config(text="Running", fg="green")
            self.process_video_frame()

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()

            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.val_status.config(text="Stopped", fg="black")
            
            # Reset camera display text
            self.video_display.configure(image="", text="Camera Offline\nClick 'Start Camera' to begin")

    def reset_prediction(self):
        self.history.clear()
        self.val_current.config(text="N/A")
        self.val_stable.config(text="N/A")
        self.val_confidence.config(text="0.00")
        self.val_hand.config(text="NO")

    def quit_app(self):
        self.stop_camera()
        self.root.destroy()

# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageGUI(root)
    root.mainloop()