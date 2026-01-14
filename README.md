🏋️‍♂️ Virtual Gym Trainer — AI-Powered Fitness Assistant

A real-time computer vision–based fitness trainer that uses a standard webcam to detect upper-body exercises, count repetitions, and provide audio + visual feedback to guide users during workouts.

This project demonstrates applied Computer Vision, Human Pose Estimation, and real-time feedback systems using Python.

🚀 Project Overview

The Virtual Gym Trainer tracks human body movements using MediaPipe Pose, computes joint angles in real time, and applies rule-based state logic to identify exercise stages and count repetitions accurately.

The system is designed for:

At-home workouts without special hardware

Real-time responsiveness

Clear visual and audio guidance

✨ Key Features

✅ Real-time pose detection using a webcam

✅ Upper-body exercise recognition and repetition counting

✅ Joint-angle computation using vector mathematics

✅ Audio feedback for reps and workout completion

✅ On-screen overlays (angles, reps, stage, exercise name)

✅ Lightweight, runs locally on CPU

🏃 Supported Exercises

Bicep Curls

Lateral Raises

Tricep Extensions

Shoulder Press

Each exercise uses custom joint-angle thresholds and stage transitions to ensure reliable repetition counting.

🧠 How It Works (High Level)
1. Pose Detection

Uses MediaPipe Pose to extract 33 body landmarks per frame

Key joints (shoulder, elbow, wrist) are tracked continuously

2. Angle Calculation

Joint angles are computed using vector dot products and trigonometry

These angles represent arm flexion, extension, or elevation

3. Repetition Logic

A finite-state approach tracks movement stages (Up / Down / Rest)

A repetition is counted only when the full motion cycle is completed

4. Feedback

Visual overlays show real-time angles and rep count

Audio cues (.wav files) indicate rep completion and workout completion

📊 Observed Results

Rep counting accuracy: ~90%+ under good lighting conditions

Latency: Near real-time feedback (< 0.2s delay)

Limitations: Reduced accuracy under poor lighting or heavy occlusion

📁 Repository Structure (Current)
Virtual-Gym-Trainer/
├─ gym_trainer1.py                 # Main execution script
├─ gym_trainers.py                 # Exercise logic & rep counting
├─ Virtual Gym Trainer Project Report 1.pdf
├─ Virtual-Gym-Trainer-AI-Powered-Fitness-Guide.pdf
├─ exercise_complete.wav           # Audio feedback
├─ workout_complete.wav            # Audio feedback
├─ good-6081.wav                   # Audio cue
├─ video1172599793.mp4             # Demo / sample video
└─ README.md

⚙️ Tech Stack

Python

OpenCV

MediaPipe (Pose)

NumPy / Math

Audio playback (WAV files)

▶️ How to Run
1. Install dependencies
pip install opencv-python mediapipe numpy

2. Run the trainer
python gym_trainer1.py

3. Usage tips

Ensure good lighting

Keep upper body clearly visible in frame

Face the camera directly for best results

🔮 Future Improvements

Form-correction using supervised learning

Adaptive thresholds per user body proportions

Lower-body exercise support

Workout history tracking and analytics

Mobile or edge-device deployment

🧩 Skills Demonstrated

Computer Vision

Human Pose Estimation

Real-Time Systems

Python Programming

Applied AI for Human-Centered Systems

📄 Reference

Detailed design, methodology, and evaluation can be found in the included project report:
“Virtual Gym Trainer – Project Report”

📌 Notes for Recruiters

This project focuses on real-time perception, logic design, and user feedback, demonstrating the ability to translate computer vision outputs into practical, interactive applications.
