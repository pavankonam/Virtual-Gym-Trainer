import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pygame
from datetime import datetime
import os

class VirtualGymTrainer:
    def __init__(self):
        # Initialize MediaPipe pose detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize pygame for audio feedback
        pygame.mixer.init()
        
        # Create sound files if they don't exist
        self.create_sound_files()
        
        # Load sound files
        try:
            self.completion_sound = pygame.mixer.Sound("exercise_complete.wav")
            self.workout_complete_sound = pygame.mixer.Sound("workout_complete.wav")
        except:
            print("Warning: Sound files not found. Audio feedback disabled.")
            self.completion_sound = None
            self.workout_complete_sound = None
        
        # Workout routine - list of exercises with their parameters
        self.workout_routine = [
            {
                "name": "Bicep Curls",
                "type": "bicep_curl",
                "reps": 5,
                "description": "Keep your elbows close to your body. Curl the weight up slowly."
            },
            {
                "name": "Lateral Raises", 
                "type": "lateral_raise",
                "reps": 2,
                "description": "Raise your arms to shoulder height. Keep slight bend in elbows."
            },
            {
                "name": "Tricep Extensions",
                "type": "tricep_extension", 
                "reps": 2,
                "description": "Keep your upper arm stationary. Only move your forearm."
            },
            {
                "name": "Shoulder Press",
                "type": "shoulder_press",
                "reps": 5,
                "description": "Press weights straight up above your head."
            }
        ]
        
        # Current workout state
        self.current_exercise_index = 0
        self.current_exercise = self.workout_routine[0]
        self.rep_counter = 0
        self.stage = None
        self.form_correct = True
        self.workout_complete = False
        self.exercise_complete = False
        self.break_start_time = None
        self.break_duration = 10  # seconds
        self.in_break = False
        
        # For tracking and smoothing
        self.angle_history = []
        self.smoothing_window = 5
        
        # Create exercise images (simple representations)
        self.create_exercise_images()
        
        # Exercise configurations
        self.exercise_configs = {
            "bicep_curl": {
                "joint_points": [self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
                               self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                               self.mp_pose.PoseLandmark.RIGHT_WRIST.value],
                "angle_down_max": 160,
                "angle_up_max": 60,
                "form_check": self.check_bicep_curl_form
            },
            "lateral_raise": {
                "joint_points": [self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                               self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                               self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                "angle_down_max": 30,
                "angle_up_max": 80,
                "form_check": self.check_lateral_raise_form
            },
            "tricep_extension": {
                "joint_points": [self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                               self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                               self.mp_pose.PoseLandmark.RIGHT_WRIST.value],
                "angle_down_max": 90,
                "angle_up_max": 160,
                "form_check": self.check_tricep_extension_form
            },
            "shoulder_press": {
                "joint_points": [self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                               self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                               self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                "angle_down_max": 90,
                "angle_up_max": 160,
                "form_check": self.check_shoulder_press_form
            }
        }
    
    def create_exercise_images(self):
        """Create simple visual representations of exercises"""
        self.exercise_images = {}
        
        # Create bicep curl image
        bicep_img = np.ones((200, 300, 3), dtype=np.uint8) * 50
        cv2.putText(bicep_img, "BICEP CURL", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Draw stick figure doing bicep curl
        cv2.line(bicep_img, (150, 60), (150, 120), (255, 255, 255), 3)  # body
        cv2.line(bicep_img, (150, 80), (120, 100), (255, 255, 255), 3)  # upper arm
        cv2.line(bicep_img, (120, 100), (110, 80), (0, 255, 0), 3)     # forearm (green for movement)
        cv2.circle(bicep_img, (150, 60), 15, (255, 255, 255), 2)       # head
        cv2.putText(bicep_img, "Curl weight up", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(bicep_img, "Keep elbows still", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.exercise_images["bicep_curl"] = bicep_img
        
        # Create lateral raise image
        lateral_img = np.ones((200, 300, 3), dtype=np.uint8) * 50
        cv2.putText(lateral_img, "LATERAL RAISE", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(lateral_img, (150, 60), (150, 120), (255, 255, 255), 3)  # body
        cv2.line(lateral_img, (150, 80), (100, 90), (0, 255, 0), 3)       # left arm raised
        cv2.line(lateral_img, (150, 80), (200, 90), (0, 255, 0), 3)       # right arm raised
        cv2.circle(lateral_img, (150, 60), 15, (255, 255, 255), 2)        # head
        cv2.putText(lateral_img, "Raise to shoulder", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(lateral_img, "height", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.exercise_images["lateral_raise"] = lateral_img
        
        # Create tricep extension image
        tricep_img = np.ones((200, 300, 3), dtype=np.uint8) * 50
        cv2.putText(tricep_img, "TRICEP EXTENSION", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(tricep_img, (150, 60), (150, 120), (255, 255, 255), 3)   # body
        cv2.line(tricep_img, (150, 80), (130, 60), (255, 255, 255), 3)    # upper arm
        cv2.line(tricep_img, (130, 60), (120, 40), (0, 255, 0), 3)       # forearm (moving part)
        cv2.circle(tricep_img, (150, 60), 15, (255, 255, 255), 2)         # head
        cv2.putText(tricep_img, "Extend forearm up", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(tricep_img, "Keep upper arm still", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.exercise_images["tricep_extension"] = tricep_img
        
        # Create shoulder press image
        shoulder_img = np.ones((200, 300, 3), dtype=np.uint8) * 50
        cv2.putText(shoulder_img, "SHOULDER PRESS", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(shoulder_img, (150, 60), (150, 120), (255, 255, 255), 3)  # body
        cv2.line(shoulder_img, (150, 80), (120, 50), (0, 255, 0), 3)       # left arm up
        cv2.line(shoulder_img, (150, 80), (180, 50), (0, 255, 0), 3)       # right arm up
        cv2.circle(shoulder_img, (150, 60), 15, (255, 255, 255), 2)        # head
        cv2.putText(shoulder_img, "Press straight up", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(shoulder_img, "Full extension", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.exercise_images["shoulder_press"] = shoulder_img
    
    def create_sound_files(self):
        """Create sound files for audio feedback"""
        try:
            # Create exercise completion sound (single beep)
            sample_rate = 22050
            duration = 0.3  # seconds
            frequency = 800  # Hz
            
            frames = int(duration * sample_rate)
            arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
            
            # Apply fade in/out to avoid clicks
            fade_frames = int(0.05 * sample_rate)
            arr[:fade_frames] *= np.linspace(0, 1, fade_frames)
            arr[-fade_frames:] *= np.linspace(1, 0, fade_frames)
            
            # Convert to 16-bit integers
            arr = (arr * 32767).astype(np.int16)
            
            # Save as WAV file
            import wave
            with wave.open("exercise_complete.wav", "w") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(arr.tobytes())
            
            # Create workout completion sound (three beeps)
            beep_duration = 0.2
            pause_duration = 0.1
            total_duration = 3 * beep_duration + 2 * pause_duration
            
            frames = int(total_duration * sample_rate)
            arr = np.zeros(frames)
            
            beep_frames = int(beep_duration * sample_rate)
            pause_frames = int(pause_duration * sample_rate)
            
            # Create three beeps
            beep = np.sin(2 * np.pi * 1000 * np.linspace(0, beep_duration, beep_frames))
            
            # Apply fade to each beep
            fade_frames = int(0.02 * sample_rate)
            beep[:fade_frames] *= np.linspace(0, 1, fade_frames)
            beep[-fade_frames:] *= np.linspace(1, 0, fade_frames)
            
            # Place beeps in the array
            arr[:beep_frames] = beep
            arr[beep_frames + pause_frames:2*beep_frames + pause_frames] = beep
            arr[2*beep_frames + 2*pause_frames:3*beep_frames + 2*pause_frames] = beep
            
            # Convert to 16-bit integers
            arr = (arr * 32767).astype(np.int16)
            
            # Save as WAV file
            with wave.open("workout_complete.wav", "w") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(arr.tobytes())
                
        except Exception as e:
            print(f"Warning: Could not create sound files: {e}")
    
    def create_completion_sound(self):
        """Removed - using saved sound files instead"""
        pass
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def smooth_angle(self, angle):
        """Apply smoothing to reduce jitter"""
        self.angle_history.append(angle)
        if len(self.angle_history) > self.smoothing_window:
            self.angle_history.pop(0)
        return np.mean(self.angle_history)
    
    def check_bicep_curl_form(self, landmarks):
        """Check bicep curl form - elbow should stay close to body, no swinging"""
        try:
            # Get key points
            shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Check 1: Elbow should stay relatively close to torso (not flaring out)
            elbow_shoulder_distance = abs(elbow.x - shoulder.x)
            max_allowed_distance = 0.15  # Adjust based on testing
            
            # Check 2: Upper arm shouldn't swing forward/backward excessively
            # Calculate if elbow is roughly in line with shoulder-hip axis
            shoulder_hip_x = (shoulder.x + hip.x) / 2
            elbow_deviation = abs(elbow.x - shoulder_hip_x)
            max_swing = 0.1
            
            # Check 3: Shoulder shouldn't elevate (no shrugging)
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            shoulder_level_diff = abs(shoulder.y - left_shoulder.y)
            max_shoulder_diff = 0.05
            
            form_good = (elbow_shoulder_distance < max_allowed_distance and 
                        elbow_deviation < max_swing and 
                        shoulder_level_diff < max_shoulder_diff)
            
            return form_good
            
        except:
            return True  # If any error in calculation, assume good form
    
    def check_lateral_raise_form(self, landmarks):
        """Check lateral raise form - arms should raise to sides, not forward"""
        try:
            # Get key points
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value] 
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            
            # Check 1: Arms should be roughly parallel to ground when raised
            shoulder_line_y = (left_shoulder.y + right_shoulder.y) / 2
            elbow_average_y = (left_elbow.y + right_elbow.y) / 2
            
            # Check 2: Elbows shouldn't go too far forward or backward
            shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2
            elbow_average_z = (left_elbow.z + right_elbow.z) / 2
            forward_backward_deviation = abs(elbow_average_z - shoulder_center_z)
            max_deviation = 0.1
            
            # Check 3: Both arms should move symmetrically
            left_arm_height = left_shoulder.y - left_elbow.y
            right_arm_height = right_shoulder.y - right_elbow.y
            symmetry_diff = abs(left_arm_height - right_arm_height)
            max_asymmetry = 0.08
            
            form_good = (forward_backward_deviation < max_deviation and 
                        symmetry_diff < max_asymmetry)
            
            return form_good
            
        except:
            return True
    
    def check_tricep_extension_form(self, landmarks):
        """Check tricep extension form - upper arm should stay stationary"""
        try:
            # Get key points
            shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Check 1: Upper arm (shoulder to elbow) should stay relatively vertical
            # Calculate angle of upper arm from vertical
            upper_arm_angle = math.atan2(elbow.x - shoulder.x, elbow.y - shoulder.y)
            upper_arm_angle_deg = abs(math.degrees(upper_arm_angle))
            max_upper_arm_deviation = 30  # degrees from vertical
            
            # Check 2: Elbow position shouldn't drift too much during movement
            # This would require tracking over time, simplified for now
            
            # Check 3: Movement should be primarily in the forearm
            # Check if wrist moves significantly more than elbow
            elbow_shoulder_dist = math.sqrt((elbow.x - shoulder.x)**2 + (elbow.y - shoulder.y)**2)
            wrist_elbow_dist = math.sqrt((wrist.x - elbow.x)**2 + (wrist.y - elbow.y)**2)
            
            # For good form, forearm should be doing most of the movement
            form_good = (upper_arm_angle_deg < max_upper_arm_deviation and
                        wrist_elbow_dist > 0.1)  # Ensure there's actual movement
            
            return form_good
            
        except:
            return True
    
    def check_shoulder_press_form(self, landmarks):
        """Check shoulder press form - press straight up, maintain core stability"""
        try:
            # Get key points
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Check 1: Wrists should be above elbows when pressing (vertical path)
            left_vertical_alignment = abs(left_wrist.x - left_elbow.x)
            right_vertical_alignment = abs(right_wrist.x - right_elbow.x)
            max_horizontal_drift = 0.08
            
            # Check 2: Both arms should move symmetrically
            left_arm_height = left_shoulder.y - left_wrist.y
            right_arm_height = right_shoulder.y - right_wrist.y
            symmetry_diff = abs(left_arm_height - right_arm_height)
            max_asymmetry = 0.1
            
            # Check 3: Shoulders should stay level (no tilting)
            shoulder_level_diff = abs(left_shoulder.y - right_shoulder.y)
            max_shoulder_tilt = 0.05
            
            form_good = (left_vertical_alignment < max_horizontal_drift and
                        right_vertical_alignment < max_horizontal_drift and
                        symmetry_diff < max_asymmetry and
                        shoulder_level_diff < max_shoulder_tilt)
            
            return form_good
            
        except:
            return True
    
    def next_exercise(self):
        """Move to the next exercise"""
        self.current_exercise_index += 1
        if self.current_exercise_index >= len(self.workout_routine):
            self.workout_complete = True
            return
        
        self.current_exercise = self.workout_routine[self.current_exercise_index]
        self.rep_counter = 0
        self.stage = None
        self.angle_history = []
        self.exercise_complete = False
    
    def start_break(self):
        """Start break between exercises"""
        self.in_break = True
        self.break_start_time = time.time()
    
    def process_frame(self, frame):
        """Process video frame and track exercise"""
        if self.workout_complete:
            return self.show_workout_complete(frame)
        
        if self.in_break:
            return self.show_break_screen(frame)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Show exercise image
        exercise_img = self.exercise_images[self.current_exercise["type"]]
        annotated_frame[10:210, 10:310] = exercise_img
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Get exercise configuration
            config = self.exercise_configs[self.current_exercise["type"]]
            landmarks = results.pose_landmarks.landmark
            
            # Calculate angle
            points = [landmarks[idx] for idx in config["joint_points"]]
            angle = self.calculate_angle(points[0], points[1], points[2])
            angle = self.smooth_angle(angle)
            
            # Check form
            self.form_correct = config["form_check"](landmarks)
            
            # Count reps
            target_reps = self.current_exercise["reps"]
            
            if angle > config["angle_down_max"]:
                self.stage = "down"
            elif angle < config["angle_up_max"] and self.stage == "down" and self.form_correct:
                self.stage = "up"
                self.rep_counter += 1
                
                if self.rep_counter >= target_reps:
                    self.exercise_complete = True
                    if self.completion_sound:
                        self.completion_sound.play()
        
        # Add UI elements
        self.add_ui_elements(annotated_frame)
        
        # Check if exercise is complete
        if self.exercise_complete and not self.in_break:
            self.start_break()
        
        return annotated_frame
    
    def add_ui_elements(self, frame):
        """Add UI elements to the frame"""
        height, width = frame.shape[:2]
        
        # Exercise info
        exercise_name = self.current_exercise["name"]
        target_reps = self.current_exercise["reps"]
        
        # Progress bar background
        cv2.rectangle(frame, (width-250, 20), (width-20, 60), (50, 50, 50), -1)
        
        # Progress bar fill
        progress = min(self.rep_counter / target_reps, 1.0)
        progress_width = int(progress * 230)
        cv2.rectangle(frame, (width-250, 20), (width-250+progress_width, 60), (0, 255, 0), -1)
        
        # Text overlays
        cv2.putText(frame, f"Exercise: {exercise_name}", 
                   (width-400, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Reps: {self.rep_counter}/{target_reps}", 
                   (width-400, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Exercise {self.current_exercise_index + 1}/{len(self.workout_routine)}", 
                   (width-400, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Form feedback
        form_text = "Good Form!" if self.form_correct else "Check Form!"
        form_color = (0, 255, 0) if self.form_correct else (0, 0, 255)
        cv2.putText(frame, form_text, 
                   (width-400, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2)
        
        # Exercise description
        description = self.current_exercise["description"]
        cv2.putText(frame, description, 
                   (20, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def show_break_screen(self, frame):
        """Show break screen between exercises"""
        if self.break_start_time is None:
            return frame
        
        elapsed = time.time() - self.break_start_time
        remaining = max(0, self.break_duration - elapsed)
        
        if remaining <= 0:
            self.in_break = False
            self.break_start_time = None
            self.next_exercise()
            return frame
        
        # Create break screen overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        
        # Add transparency
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Break text
        cv2.putText(frame, "BREAK TIME!", 
                   (frame.shape[1]//2 - 120, frame.shape[0]//2 - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        
        cv2.putText(frame, f"Next: {self.workout_routine[self.current_exercise_index + 1]['name'] if self.current_exercise_index + 1 < len(self.workout_routine) else 'Workout Complete'}", 
                   (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Time remaining: {int(remaining)}s", 
                   (frame.shape[1]//2 - 100, frame.shape[0]//2 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def show_workout_complete(self, frame):
        """Show workout completion screen"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 100, 0), -1)
        
        alpha = 0.9
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Completion text
        cv2.putText(frame, "WORKOUT COMPLETE!", 
                   (frame.shape[1]//2 - 200, frame.shape[0]//2 - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        cv2.putText(frame, "Great job! You completed all exercises!", 
                   (frame.shape[1]//2 - 180, frame.shape[0]//2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

def main():
    # Initialize the virtual trainer
    trainer = VirtualGymTrainer()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set window size
    cv2.namedWindow('Virtual Gym Trainer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Virtual Gym Trainer', 1200, 800)
    
    workout_complete_start = None
    
    print("Starting Virtual Gym Trainer!")
    print("Workout includes:")
    for i, exercise in enumerate(trainer.workout_routine, 1):
        print(f"{i}. {exercise['name']} - {exercise['reps']} reps")
    print("\nPress 'q' to quit early")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process the frame
        annotated_frame = trainer.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Virtual Gym Trainer', annotated_frame)
        
        # Handle workout completion
        if trainer.workout_complete:
            if workout_complete_start is None:
                workout_complete_start = time.time()
                if trainer.workout_complete_sound:
                    trainer.workout_complete_sound.play()
                print("🎉 WORKOUT COMPLETE! Great job!")
            elif time.time() - workout_complete_start >= 10:  # Show for 10 seconds
                break
        
        # Break on 'q' press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Virtual Gym Trainer session ended. Keep up the great work!")

if __name__ == "__main__":
    main()