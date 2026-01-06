Chaos Tracker: Real-Time Non-Linear Dynamics Analysis

A Computer Vision-based data acquisition system for analyzing the chaotic evolution of a double pendulum in real-time.

Scientific Context

The double pendulum is a canonical example of a Hamiltonian system exhibiting Deterministic Chaos. While governed by simple Lagrangian mechanics, its trajectory is highly sensitive to initial conditions (the Butterfly Effect).

This project moves beyond numerical simulation to experimental measurement. Using high-speed computer vision, it extracts kinematic data from a physical pendulum to visualise:

Phase Space Topology: The evolution of the system in configuration space ($\theta_1$ vs $\theta_2$).

Ergodicity: The transition from quasi-periodic motion (low energy) to chaotic mixing (high energy).

Time-Series Analysis: Real-time tracking of angular displacement for stability analysis.

Key Features

Robust Optical Tracking:

Utilises HSV Colour Space segmentation with morphological filtering (Erosion/Dilation) to track markers at high velocities where standard fiducial markers (ArUco) fail due to motion blur.

Implements a deque-based smoothing filter (Moving Average) to reduce sensor quantisation noise and jitter.

Real-Time "Research Dashboard":

Phase Space Plot: Dynamic rendering of the Configuration Space trajectory.

Time Series Graph: Live scrolling plot of angular evolution.

Augmented Reality Overlay: Visualises the "Skeleton" and infinite motion trails ($O(1)$ rendering complexity).

Hardware Integration:

Automated camera index scanning for robust connection with DroidCam/USB sources.

Optimised for 60 FPS data ingestion on standard hardware.

 Installation

Clone the Repository

git clone [https://github.com/YourUsername/Double-Pendulum-Chaos-Tracker.git](https://github.com/YourUsername/Double-Pendulum-Chaos-Tracker.git)
cd Double-Pendulum-Chaos-Tracker


Install Dependencies

pip install -r requirements.txt


Usage

Prepare the Setup:

Hardware: A physical double pendulum with a Neon Green sticker on the middle joint and a Neon Pink sticker on the bottom weight.

Camera: Connect your phone via DroidCam (USB mode recommended for 60 FPS) or use a webcam.

Run the Tracker:

python chaos_tracker.py


Calibration (Critical Step):
The system initiates a 3-step click calibration to map the physical dimensions:

Step 1: Click the Top Pivot (Nail/Bearing) on the camera feed.

Step 2: Click the Middle Joint (Green Marker).

Step 3: Click the Bottom Weight (Pink Marker).

The system applies vector subtraction to track the true mechanical joint based on the visual marker position.

Controls:

L: Toggle Recording (Saves .avi to local folder).

C: Clear all trails and graphs (Reset).

Z: Toggle Zoom (Switch between "Chaos" view and "Small Angle" view).

Q: Quit.

Mathematical Framework

The system maps Cartesian pixel coordinates $(x, y)$ to Generalized Coordinates ($\theta_1, \theta_2$) using vector geometry:

Given the pivot $P_0$, middle joint $P_1$, and end mass $P_2$:

$$\theta_1 = \arctan2(P_{1y} - P_{0y}, P_{1x} - P_{0x})$$
$$\theta_2 = \arctan2(P_{2y} - P_{1y}, P_{2x} - P_{1x})$$

These angles form the basis for the real-time Phase Space reconstruction.

Future Work (Research Roadmap)

Dynamic Analysis: Comparison of experimental data with a numerical Runge-Kutta 4 (RK4) simulation to measure the Lyapunov Time (horizon of predictability).

Geometric Stability: Implementation of KCC Theory to calculate the Deviation Curvature Tensor from the time-series data.

Friction Modeling: Curve-fitting energy decay to determine the coefficients of viscous vs. Coulomb damping.
