import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pygame

class VirtualGymTrainer:
    def __init__(self, target_reps=20, exercise_type="bicep_curl"):
        # Initialize MediaPipe pose detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Exercise parameters
        self.target_reps = target_reps
        self.exercise_type = exercise_type
        self.rep_counter = 0
        self.stage = None  # "down" or "up"
        self.form_correct = True
        
        # For tracking and smoothing
        self.angle_history = []
        self.smoothing_window = 5
        
        # Initialize pygame for audio feedback
        pygame.mixer.init()
        self.sound_complete = pygame.mixer.Sound("good-6081.wav")  # You'll need to create/download this sound file
        
        # Exercise thresholds (angles in degrees)
        self.exercise_configs = {
            "bicep_curl": {
                "joint_points": [self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
                               self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                               self.mp_pose.PoseLandmark.RIGHT_WRIST.value],
                "angle_down_max": 160,  # Arm extended
                "angle_up_max": 60,     # Arm curled
                "form_check": self.check_bicep_curl_form
            },
            "lateral_raise": {
                "joint_points": [self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                               self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                               self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                "angle_down_max": 20,   # Arm down
                "angle_up_max": 90,     # Arm raised to shoulder level
                "form_check": self.check_lateral_raise_form
            },
            "tricep_extension": {
                "joint_points": [self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                               self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                               self.mp_pose.PoseLandmark.RIGHT_WRIST.value],
                "angle_down_max": 90,   # Arm bent
                "angle_up_max": 160,    # Arm extended down
                "form_check": self.check_tricep_extension_form
            }
        }
    
    def calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points (in degrees)
        """
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        # Ensure the angle is between 0 and 180
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def smooth_angle(self, angle):
        """
        Apply smoothing to reduce jitter in angle measurements
        """
        self.angle_history.append(angle)
        if len(self.angle_history) > self.smoothing_window:
            self.angle_history.pop(0)
        return np.mean(self.angle_history)
    
    def check_bicep_curl_form(self, landmarks):
        """
        Check if bicep curl form is correct
        """
        # Check if shoulder is stable (not moving up)
        shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Check if elbow stays close to body
        shoulder_pos = np.array([shoulder.x, shoulder.z])
        elbow_pos = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].z])
        
        # This is a simplified check - in reality you'd want more sophisticated analysis
        # For a real application, you'd check for swinging, elbow position, etc.
        return True  # Placeholder - implement real checks based on your criteria
    
    def check_lateral_raise_form(self, landmarks):
        """
        Check if lateral raise form is correct
        """
        # Check if back is straight
        # Check if elbow is slightly bent (not locked)
        # Check if motion is lateral, not forward
        return True  # Placeholder - implement real checks
    
    def check_tricep_extension_form(self, landmarks):
        """
        Check if tricep extension form is correct
        """
        # Check if upper arm is stationary
        # Check if motion is isolated to forearm
        return True  # Placeholder - implement real checks
    
    def process_frame(self, frame):
        """
        Process a video frame, track exercise, and return annotated frame
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and get pose landmarks
        results = self.pose.process(frame_rgb)
        
        # Draw landmarks on the frame
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Get exercise config
            config = self.exercise_configs[self.exercise_type]
            
            # Get landmarks for the angles we need to measure
            landmarks = results.pose_landmarks.landmark
            
            # Extract points for angle calculation
            points = [landmarks[idx] for idx in config["joint_points"]]
            
            # Calculate and smooth the angle
            angle = self.calculate_angle(points[0], points[1], points[2])
            angle = self.smooth_angle(angle)
            
            # Check form
            self.form_correct = config["form_check"](landmarks)
            
            # Count reps based on angle thresholds
            if angle > config["angle_down_max"]:
                self.stage = "down"
            elif angle < config["angle_up_max"] and self.stage == "down" and self.form_correct:
                self.stage = "up"
                self.rep_counter += 1
                print(f"Rep {self.rep_counter} completed!")
                
                # Check if we've reached the target
                if self.rep_counter >= self.target_reps:
                    print("Exercise complete!")
                    self.sound_complete.play()
            
            # Display info on frame
            cv2.putText(annotated_frame, f"Reps: {self.rep_counter}/{self.target_reps}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(annotated_frame, f"Angle: {int(angle)}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Form feedback
            form_text = "Good Form" if self.form_correct else "Check Form"
            form_color = (0, 255, 0) if self.form_correct else (0, 0, 255)
            cv2.putText(annotated_frame, form_text, 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, form_color, 2, cv2.LINE_AA)
            
        return annotated_frame
    
    def reset(self):
        """
        Reset the exercise counter
        """
        self.rep_counter = 0
        self.stage = None
        self.angle_history = []

def main():
    # Create sound file (a simple beep)
    #pygame.mixer.init()
   # freq = 440  # Hz
    #duration = 500  # ms
    #pygame.mixer.Sound(np.sin(2*np.pi*np.arange(44100*duration/1000)/44100*freq).astype(np.float32)).save("good-6081.wav")
    
    # Initialize the virtual trainer
    exercise_type = "bicep_curl"  # Options: "bicep_curl", "lateral_raise", "tricep_extension"
    target_reps = 20
    trainer = VirtualGymTrainer(target_reps=target_reps, exercise_type=exercise_type)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        annotated_frame = trainer.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Virtual Gym Trainer', annotated_frame)
        
        # Break loop on 'q' press or when target reps reached
        if cv2.waitKey(10) & 0xFF == ord('q') or trainer.rep_counter >= target_reps:
            # Wait a moment if exercise is complete
            if trainer.rep_counter >= target_reps:
                time.sleep(2)  # Give time to see completion and hear sound
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()