# Chaos Tracker: Real-Time Double Pendulum Analysis

A computer vision system that tracks a physical double pendulum and visualizes its chaotic motion in real-time.

![Dashboard Demo](dashboard_log.gif)

## What This Does

The double pendulum is one of the simplest systems that exhibits **deterministic chaos** — it follows clear physical laws, but tiny differences in starting conditions lead to completely different outcomes (the famous Butterfly Effect). While it's easy to simulate on a computer, I wanted to actually *measure* it happening in real life.

This project uses a webcam to track colored markers on a real pendulum and generates live visualizations:

- **Phase Space Plot**: Shows how the system evolves through angular velocity space (ω₁ vs ω₂) with adaptive auto-scaling axes
- **Time Series Graph**: Tracks how each angle changes over time
- **Motion Trails**: Beautiful overlay showing where the pendulum has been

## Why Computer Vision?

I initially tried using ArUco markers (those QR-code-like tags), but they completely fail when the pendulum moves fast – everything just becomes a blurry mess. So instead, I went with bright colored stickers tracked using HSV color space. Way more robust.

The tracking pipeline includes:
- Morphological filtering (erosion/dilation) to clean up the detection
- A moving average smoothing filter to reduce jitter from the camera
- Automatic camera connection (it'll search through your USB devices to find the right one)

## Features

**Real-Time Dashboard**
- Live phase space trajectory in ω₁ vs ω₂ — the true dynamical state of the system
- Adaptive axis scaling that auto-ranges to the observed velocities, so the plot always fills the canvas
- Scrolling time series plot of θ₁ and θ₂
- Augmented reality overlay showing the pendulum "skeleton" and infinite trails
- Efficient rendering that won't lag your computer

**Robust Tracking**
- Works at high velocities where other methods fail
- Handles lighting variations reasonably well
- Built-in position smoothing to deal with camera noise
- Timestamped dt for physically correct angular velocity (rad/s) regardless of frame rate fluctuation

**Hardware Integration**
- Works with WebCam (phone as webcam) or regular USB cameras
- Optimized for 60 FPS capture
- Auto-detects camera index and matches recording FPS to actual capture rate

## Installation

### Clone the Repository
```bash
git clone https://github.com/kazi-kame/Double-Pendulum-Chaos-Tracker.git
cd Chaos-Tracker
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Setup Your Pendulum
You'll need:
- A physical double pendulum
- A **neon green** sticker on the middle joint
- A **neon pink** sticker on the bottom weight

### Connect Your Camera
Either use WebCam in USB mode (recommended for 60 FPS) or just plug in a webcam.

### Run the Tracker
```bash
python chaos_tracker.py
```

### Calibration
The first time you run it, you need to click three points so the system knows where everything is:

1. **Step 1**: Click the top pivot (where the pendulum hangs from)
2. **Step 2**: Click the middle joint (green marker)
3. **Step 3**: Click the bottom weight (pink marker)

The system uses these to calculate offsets – it needs to know where the actual mechanical joints are relative to where the stickers appear on camera.

### Controls
- **L**: Start/Stop recording (saves as .avi)
- **C**: Clear all trails and graphs
- **Q**: Quit

## The Math Behind It

The tracker converts pixel coordinates $(x,y)$ from the camera into the angles $(\theta_1, \theta_2)$ of the pendulum:

Given the pivot $P_0$, middle joint $P_1$, and end mass $P_2$:

$$\theta_1 = \arctan2(P_{1x} - P_{0x},\ P_{1y} - P_{0y})$$

$$\theta_2 = \arctan2(P_{2x} - P_{1x},\ P_{2y} - P_{1y})$$

Angular velocities are computed via finite difference using a real wall-clock timestamp each frame:

$$\omega_i = \frac{\Delta\theta_i}{\Delta t}$$

where $\Delta t$ is measured with `time.perf_counter()` rather than assumed from nominal FPS. The $\Delta\theta$ is wrapped to $[-\pi, \pi]$ to handle the atan2 discontinuity. These ω values are smoothed with a short rolling average and plotted in the phase space diagram — each point represents the instantaneous angular velocity state of the system.

The phase space axes auto-scale each frame based on the running maximum |ω| observed over the last ~10 seconds, so the plot always uses the full canvas regardless of how energetically the pendulum is swinging.

## What I Learned

This project taught me a lot about the gap between theory and practice. The equations are simple, but getting clean data from a real physical system is surprisingly hard. Camera noise, lighting changes, motion blur, vibrations – there are so many things that can mess up your measurements.

The most interesting part was watching the phase space plot fill in during long runs. In the chaos regime it eventually covers the whole space (ergodic behavior), while in small-angle oscillations you get nice closed loops — visible even in the ω₁ vs ω₂ plane as tight elliptical orbits.

## Future Ideas

Things I want to add eventually:

- **Lyapunov Exponent Calculation**: Quantify exactly how chaotic the motion is
- **RK4 Simulation Comparison**: Run a numerical simulation alongside the real data to see how long they stay synchronized
- **Energy Analysis**: Track the total energy over time and fit damping coefficients
- **Deviation Curvature**: Apply KCC theory to measure geometric stability
- **Kalman Filter**: Replace the moving average with a proper state estimator for better noise rejection at high velocities

## Acknowledgments

This started as a pastime coz I was bored. Big thanks to my classmate Devang Rathod for helping me out.
