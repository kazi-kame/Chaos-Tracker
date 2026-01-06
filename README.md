Chaos Tracker: Real-Time Double Pendulum Analysis
A computer vision system that tracks a physical double pendulum and visualizes its chaotic motion in real-time.
What This Does
The double pendulum is one of the simplest systems that exhibits deterministic chaos – it follows clear physical laws, but tiny differences in starting conditions lead to completely different outcomes (the famous Butterfly Effect). While it's easy to simulate on a computer, I wanted to actually measure it happening in real life.
This project uses a webcam to track colored markers on a real pendulum and generates live visualizations:

Phase Space Plot: Shows how the system evolves through different angle combinations (θ₁ vs θ₂)
Time Series Graph: Tracks how each angle changes over time
Motion Trails: Beautiful overlay showing where the pendulum has been

Why Computer Vision?
I initially tried using ArUco markers (those QR-code-like tags), but they completely fail when the pendulum moves fast – everything just becomes a blurry mess. So instead, I went with bright colored stickers tracked using HSV color space. Way more robust.
The tracking pipeline includes:

Morphological filtering (erosion/dilation) to clean up the detection
A moving average smoothing filter to reduce jitter from the camera
Automatic camera connection (it'll search through your USB devices to find the right one)

Features
Real-Time Dashboard

Live phase space trajectory rendering
Scrolling time series plot
Augmented reality overlay showing the pendulum "skeleton" and infinite trails
Efficient rendering that won't lag your computer

Robust Tracking

Works at high velocities where other methods fail
Handles lighting variations reasonably well
Built-in position smoothing to deal with camera noise

Hardware Integration

Works with DroidCam (phone as webcam) or regular USB cameras
Optimized for 60 FPS capture
Auto-detects camera index

Installation
Clone the Repository
bashgit clone https://github.com/YourUsername/Double-Pendulum-Chaos-Tracker.git
cd Double-Pendulum-Chaos-Tracker
Install Dependencies
bashpip install -r requirements.txt
Usage
Setup Your Pendulum
You'll need:

A physical double pendulum (I used wooden rods with bearings)
A neon green sticker on the middle joint
A neon pink sticker on the bottom weight

Connect Your Camera
Either use DroidCam in USB mode (recommended for 60 FPS) or just plug in a webcam.
Run the Tracker
bashpython chaos_tracker.py
Calibration
The first time you run it, you need to click three points so the system knows where everything is:

Step 1: Click the top pivot (where the pendulum hangs from)
Step 2: Click the middle joint (green marker)
Step 3: Click the bottom weight (pink marker)

The system uses these to calculate offsets – it needs to know where the actual mechanical joints are relative to where the stickers appear on camera.
Controls

L: Start/Stop recording (saves as .avi)
C: Clear all trails and graphs
Z: Toggle between "Chaos" zoom and "Small Angle" zoom
Q: Quit

The Math Behind It
The tracker converts pixel coordinates (x,y) from the camera into the actual angles (θ₁,θ₂) of the pendulum:
Given the pivot P₀, middle joint P₁, and end mass P₂:
θ₁ = arctan2(P₁y - P₀y, P₁x - P₀x)
θ₂ = arctan2(P₂y - P₁y, P₂x - P₁x)
These angles are what get plotted in the phase space diagram – each point represents a specific configuration of the pendulum.
What I Learned
This project taught me a lot about the gap between theory and practice. The equations are simple, but getting clean data from a real physical system is surprisingly hard. Camera noise, lighting changes, motion blur, vibrations – there are so many things that can mess up your measurements.
The most interesting part was watching the phase space plot fill in during long runs. In the "chaos" regime, it eventually fills the entire space (ergodic behavior), while in small-angle oscillations you just get these nice closed loops.
Future Ideas
Things I want to add eventually:

Lyapunov Exponent Calculation: Quantify exactly how chaotic the motion is
RK4 Simulation Comparison: Run a numerical simulation alongside the real data to see how long they stay synchronized
Energy Analysis: Track the total energy over time and fit damping coefficients
Deviation Curvature: Apply KCC theory to measure geometric stability
