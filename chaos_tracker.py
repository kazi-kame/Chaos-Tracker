import cv2
import numpy as np
import math
import os
from collections import deque

# Camera Configuration
MIRROR_MODE = False

# Dashboard Layout
DASHBOARD_WIDTH = 1280
DASHBOARD_HEIGHT = 720
GRAPH_WIDTH = 500
GRAPH_HEIGHT_TOP = 360
GRAPH_HEIGHT_BOT = 360
TIME_WINDOW = 200

# HSV Color Detection Ranges
PINK_LOWER = np.array([140, 50, 50])
PINK_UPPER = np.array([179, 255, 255])
RANGE_PINK = (PINK_LOWER, PINK_UPPER)

GREEN_LOWER = np.array([35, 50, 50])
GREEN_UPPER = np.array([85, 255, 255])
RANGE_GREEN = (GREEN_LOWER, GREEN_UPPER)

# Detection Parameters
SMOOTH_WINDOW = 5
MIN_CONTOUR_AREA = 100

# UI Color Scheme
COLOR_BG = (0, 0, 0)
COLOR_TOP = (255, 255, 0)
COLOR_BOT = (255, 0, 255)
COLOR_AXIS = (0, 150, 0)
COLOR_GRID = (0, 60, 0)
COLOR_TEXT = (200, 255, 200)
COLOR_HEAD = (255, 255, 255)
COLOR_PHASE = (0, 255, 255)
COLOR_SKEL = (50, 200, 50) 

# State Variables
calibration_step = 0
offsets = {'pivot': None, 'mid_offset': (0,0), 'bot_offset': (0,0)}

recording = False
out = None
file_counter = 1

prev_pt_0 = None
prev_pt_1 = None
prev_theta = None

pos_buffer_0 = deque(maxlen=SMOOTH_WINDOW)
pos_buffer_1 = deque(maxlen=SMOOTH_WINDOW)

time_theta_1 = deque(maxlen=TIME_WINDOW)
time_theta_2 = deque(maxlen=TIME_WINDOW)

# Omega tracking state
prev_theta1 = None
prev_theta2 = None
omega1 = 0.0
omega2 = 0.0
OMEGA_SMOOTH = 5
omega1_buffer = deque(maxlen=OMEGA_SMOOTH)
omega2_buffer = deque(maxlen=OMEGA_SMOOTH)
prev_time = None

# Adaptive scaling state — tracks running max |omega| seen
OMEGA_SCALE_WINDOW = 300  # frames to consider for max
omega_max_buffer = deque(maxlen=OMEGA_SCALE_WINDOW)
adaptive_scale = 1.0      # rad/s, grows to fit data

print("\n" + "="*60)
print("DOUBLE PENDULUM CHAOS TRACKER - HSV Color Detection")
print("="*60)
print("Camera Detection:")
print("  Searching for: WebCam USB connection")
print("="*60)

import time

cap = None
ret = False

print("Searching for WebCam USB...")
for idx in [1, 2, 3, 4]:
    print(f"Trying camera index {idx}...")
    test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    test_cap.set(cv2.CAP_PROP_FPS, 60)
    
    time.sleep(0.3)
    test_ret, test_frame = test_cap.read()
    
    if test_ret and test_frame is not None:
        if np.std(test_frame) > 10:
            print(f"Found valid camera at index {idx}")
            cap = test_cap
            ret = test_ret
            CAMERA_INDEX = idx
            break
    
    test_cap.release()

if not ret or cap is None:
    print("\nError: Could not find WebCam!")
    exit()
    
actual_fps = cap.get(cv2.CAP_PROP_FPS)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\nWebCam USB Connected!")
print(f"Resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")

ret, test_frame = cap.read()
if not ret:
    print("Error reading frame from WebCam")
    cap.release()
    exit()

SRC_H, SRC_W = test_frame.shape[:2]
DISP_W = DASHBOARD_WIDTH - GRAPH_WIDTH
DISP_H = DASHBOARD_HEIGHT

trail_canvas = np.zeros_like(test_frame)
phase_canvas = np.zeros((GRAPH_HEIGHT_TOP, GRAPH_WIDTH, 3), dtype=np.uint8)

init_phase_done = False

def smooth_position(buffer):
    """Average positions in buffer for motion smoothing."""
    if len(buffer) == 0:
        return None
    x_vals = [p[0] for p in buffer]
    y_vals = [p[1] for p in buffer]
    return (int(np.mean(x_vals)), int(np.mean(y_vals)))

def detect_color_center(hsv_frame, color_range, min_area=MIN_CONTOUR_AREA):
    """Detect colored marker and return center coordinates."""
    lower, upper = color_range
    mask = cv2.inRange(hsv_frame, lower, upper)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, mask
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < min_area:
        return None, mask
    
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, mask
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy), mask

def draw_phase_grid(img):
    """Initialize phase space plot with grid and labels."""
    img[:] = 0
    
    for i in range(0, GRAPH_WIDTH, 50):
        cv2.line(img, (i, 0), (i, GRAPH_HEIGHT_TOP), COLOR_GRID, 1)
    for i in range(0, GRAPH_HEIGHT_TOP, 50):
        cv2.line(img, (0, i), (GRAPH_WIDTH, i), COLOR_GRID, 1)
    
    cx, cy = GRAPH_WIDTH // 2, GRAPH_HEIGHT_TOP // 2
    cv2.line(img, (0, cy), (GRAPH_WIDTH, cy), COLOR_AXIS, 2)
    cv2.line(img, (cx, 0), (cx, GRAPH_HEIGHT_TOP), COLOR_AXIS, 2)
    
    cv2.putText(img, "PHASE SPACE: \u03c9\u2081 vs \u03c9\u2082", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, COLOR_TEXT, 1)
    cv2.putText(img, "X: Omega 1 | Y: Omega 2", (10, GRAPH_HEIGHT_TOP - 10), 
                cv2.FONT_HERSHEY_PLAIN, 0.8, COLOR_AXIS, 1)

def draw_time_grid(img):
    """Initialize time series plot with grid and labels."""
    for i in range(0, GRAPH_WIDTH, 50):
        cv2.line(img, (i, 0), (i, GRAPH_HEIGHT_BOT), COLOR_GRID, 1)
    for i in range(0, GRAPH_HEIGHT_BOT, 50):
        cv2.line(img, (0, i), (GRAPH_WIDTH, i), COLOR_GRID, 1)
    
    cy = GRAPH_HEIGHT_BOT // 2
    cv2.line(img, (0, cy), (GRAPH_WIDTH, cy), COLOR_AXIS, 2)
    
    cv2.putText(img, "TIME SERIES", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, COLOR_TEXT, 1)
    cv2.putText(img, "+PI", (GRAPH_WIDTH - 40, 20), cv2.FONT_HERSHEY_PLAIN, 0.8, COLOR_AXIS, 1)
    cv2.putText(img, "-PI", (GRAPH_WIDTH - 40, GRAPH_HEIGHT_BOT - 10), 
                cv2.FONT_HERSHEY_PLAIN, 0.8, COLOR_AXIS, 1)

def draw_time_series(img, data1, data2):
    """Render angle history over time."""
    if len(data1) < 2: 
        return
    
    scale_y = (GRAPH_HEIGHT_BOT / 2) / math.pi
    cy = GRAPH_HEIGHT_BOT // 2
    dx = GRAPH_WIDTH / TIME_WINDOW

    pts1 = []
    pts2 = []
    for i, (t1, t2) in enumerate(zip(data1, data2)):
        x = int(i * dx)
        y1 = int(cy - (t1 * scale_y))
        y2 = int(cy - (t2 * scale_y))
        pts1.append((x, y1))
        pts2.append((x, y2))

    cv2.polylines(img, [np.array(pts1)], False, COLOR_TOP, 1)
    cv2.polylines(img, [np.array(pts2)], False, COLOR_BOT, 1)

def mouse_callback(event, x, y, flags, param):
    """Handle calibration point selection."""
    global calibration_step, offsets
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if x > GRAPH_WIDTH:
            click_x_disp = x - GRAPH_WIDTH
            click_y_disp = y
            
            src_x = int(click_x_disp * (SRC_W / DISP_W))
            src_y = int(click_y_disp * (SRC_H / DISP_H))
            
            if calibration_step == 0:
                offsets['pivot'] = (src_x, src_y)
                print(f"Pivot set: {src_x}, {src_y}")
                calibration_step = 1
            elif calibration_step == 1 and current_color_0_pos:
                offsets['mid_offset'] = (src_x - current_color_0_pos[0], 
                                        src_y - current_color_0_pos[1])
                print("Middle offset set.")
                calibration_step = 2
            elif calibration_step == 2 and current_color_1_pos:
                offsets['bot_offset'] = (src_x - current_color_1_pos[0], 
                                        src_y - current_color_1_pos[1])
                print("Bottom offset set.")
                calibration_step = 3

cv2.namedWindow("Chaos Dashboard")
cv2.setMouseCallback("Chaos Dashboard", mouse_callback)

print("\n" + "="*60)
print("DOUBLE PENDULUM CHAOS TRACKER - HSV Color Detection")
print("="*60)
print("Controls:")
print("  L - Start/Stop Recording")
print("  C - Clear trails and reset graphs")
print("  Q - Quit")
print("\nCalibration Steps:")
print("  1. Click on the top pivot point")
print("  2. Click on the middle joint (GREEN marker)")
print("  3. Click on the bottom weight (PINK marker)")
print("\nColor Detection:")
print("  GREEN marker - Middle joint (HSV: 35-85)")
print("  PINK marker - Bottom weight (HSV: 140-179)")
print("\nVideo Source:")
print("  Using OBS Virtual Camera or similar device")
print("="*60 + "\n")

current_color_0_pos = None
current_color_1_pos = None

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Lost camera connection!")
        break
    
    if MIRROR_MODE: 
        frame = cv2.flip(frame, 1)

    display_frame = frame.copy()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_center, green_mask = detect_color_center(hsv_frame, RANGE_GREEN)
    pink_center, pink_mask = detect_color_center(hsv_frame, RANGE_PINK)
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    combined_mask = cv2.bitwise_or(green_mask, pink_mask)
    
    color_regions = cv2.bitwise_and(frame, frame, mask=combined_mask)
    gray_regions = cv2.bitwise_and(gray_frame_bgr, gray_frame_bgr, mask=cv2.bitwise_not(combined_mask))
    
    display_frame = cv2.add(gray_regions, color_regions)
    
    current_color_0_pos = green_center
    current_color_1_pos = pink_center
    
    if green_center:
        pos_buffer_0.append(green_center)
    if pink_center:
        pos_buffer_1.append(pink_center)
    
    smooth_pos_0 = smooth_position(pos_buffer_0)
    smooth_pos_1 = smooth_position(pos_buffer_1)
    
    if current_color_0_pos:
        cv2.circle(display_frame, current_color_0_pos, 8, (0, 255, 0), 2)
        cv2.putText(display_frame, "GREEN", (current_color_0_pos[0]+12, current_color_0_pos[1]), 
                   cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    
    if current_color_1_pos:
        cv2.circle(display_frame, current_color_1_pos, 8, (255, 0, 255), 2)
        cv2.putText(display_frame, "PINK", (current_color_1_pos[0]+12, current_color_1_pos[1]), 
                   cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    if not init_phase_done:
        draw_phase_grid(phase_canvas)
        init_phase_done = True

    time_canvas = np.zeros((GRAPH_HEIGHT_BOT, GRAPH_WIDTH, 3), dtype=np.uint8)
    draw_time_grid(time_canvas)

    current_phase_head = None

    if calibration_step == 3:
        curr_pt_0 = None
        if smooth_pos_0:
            curr_pt_0 = (smooth_pos_0[0] + offsets['mid_offset'][0], 
                         smooth_pos_0[1] + offsets['mid_offset'][1])

        curr_pt_1 = None
        if smooth_pos_1:
            curr_pt_1 = (smooth_pos_1[0] + offsets['bot_offset'][0], 
                         smooth_pos_1[1] + offsets['bot_offset'][1])

        if offsets['pivot'] and curr_pt_0:
            cv2.line(display_frame, offsets['pivot'], curr_pt_0, COLOR_SKEL, 2)
        if curr_pt_0 and curr_pt_1:
            cv2.line(display_frame, curr_pt_0, curr_pt_1, COLOR_SKEL, 2)

        if prev_pt_0 and curr_pt_0:
            cv2.line(trail_canvas, prev_pt_0, curr_pt_0, COLOR_TOP, 2)
        if prev_pt_1 and curr_pt_1:
            cv2.line(trail_canvas, prev_pt_1, curr_pt_1, COLOR_BOT, 2)
        
        if curr_pt_0: 
            prev_pt_0 = curr_pt_0
        if curr_pt_1: 
            prev_pt_1 = curr_pt_1

        if offsets['pivot'] and curr_pt_0 and curr_pt_1:
            dx1 = curr_pt_0[0] - offsets['pivot'][0]
            dy1 = curr_pt_0[1] - offsets['pivot'][1]
            theta1 = math.atan2(dx1, dy1) 

            dx2 = curr_pt_1[0] - curr_pt_0[0]
            dy2 = curr_pt_1[1] - curr_pt_0[1]
            theta2 = math.atan2(dx2, dy2)

            time_theta_1.append(theta1)
            time_theta_2.append(theta2)

            # Timestamped dt
            now = time.perf_counter()
            if prev_time is not None:
                dt = now - prev_time
            else:
                dt = 1.0 / max(actual_fps, 1)
            prev_time = now

            # Compute angular velocities via finite difference (rad/s), smoothed
            if prev_theta1 is not None and prev_theta2 is not None and dt > 0:
                raw_omega1 = theta1 - prev_theta1
                raw_omega2 = theta2 - prev_theta2
                # Wrap to [-pi, pi] to handle angle discontinuities
                raw_omega1 = (raw_omega1 + math.pi) % (2 * math.pi) - math.pi
                raw_omega2 = (raw_omega2 + math.pi) % (2 * math.pi) - math.pi
                omega1_buffer.append(raw_omega1 / dt)
                omega2_buffer.append(raw_omega2 / dt)

            prev_theta1 = theta1
            prev_theta2 = theta2

            if len(omega1_buffer) > 0:
                omega1 = float(np.mean(omega1_buffer))
                omega2 = float(np.mean(omega2_buffer))

                # Adaptive scaling: track running max and pad by 20%
                omega_max_buffer.append(max(abs(omega1), abs(omega2)))
                adaptive_scale = max(omega_max_buffer) * 1.2
                adaptive_scale = max(adaptive_scale, 0.5)  # floor to avoid divide-by-zero at rest

                scale_factor = (GRAPH_WIDTH / 2) / adaptive_scale

                gx = int((GRAPH_WIDTH // 2) + (omega1 * scale_factor))
                gy = int((GRAPH_HEIGHT_TOP // 2) - (omega2 * scale_factor))

                gx = np.clip(gx, 0, GRAPH_WIDTH - 1)
                gy = np.clip(gy, 0, GRAPH_HEIGHT_TOP - 1)

                if prev_theta:
                    dist = math.hypot(gx - prev_theta[0], gy - prev_theta[1])
                    if dist < GRAPH_WIDTH / 3:
                        cv2.line(phase_canvas, prev_theta, (gx, gy), COLOR_PHASE, 1)
                
                prev_theta = (gx, gy)
                current_phase_head = (gx, gy)

                # Overlay current scale on phase canvas so you know the axis range
                scale_label = f"scale: +/-{adaptive_scale:.1f} rad/s"
                cv2.putText(phase_canvas, scale_label, (GRAPH_WIDTH - 200, 20),
                            cv2.FONT_HERSHEY_PLAIN, 0.9, COLOR_TEXT, 1)

    draw_time_series(time_canvas, time_theta_1, time_theta_2)

    full_source_display = cv2.add(display_frame, trail_canvas)
    cam_display = cv2.resize(full_source_display, (DISP_W, DISP_H))
    
    phase_display = phase_canvas.copy()
    if current_phase_head:
        cv2.circle(phase_display, current_phase_head, 4, COLOR_HEAD, -1)

    left_panel = np.vstack((phase_display, time_canvas))
    dashboard = np.hstack((left_panel, cam_display))

    font = cv2.FONT_HERSHEY_DUPLEX
    if calibration_step < 3:
        cv2.putText(dashboard, f"CALIBRATION: Step {calibration_step + 1} of 3", 
                   (GRAPH_WIDTH + 30, 50), font, 0.8, (0,0,255), 2)
        
        if calibration_step == 0: 
            msg = "CLICK TOP PIVOT"
        elif calibration_step == 1: 
            msg = "CLICK MID JOINT (GREEN)"
        elif calibration_step == 2: 
            msg = "CLICK BOT WEIGHT (PINK)"
        
        cv2.putText(dashboard, msg, (GRAPH_WIDTH + 30, 80), font, 0.7, (0, 0, 255), 2)
    
    status_y = DASHBOARD_HEIGHT - 60
    if smooth_pos_0:
        cv2.putText(dashboard, "GREEN: OK", (GRAPH_WIDTH + 30, status_y), 
                   font, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(dashboard, "GREEN: LOST", (GRAPH_WIDTH + 30, status_y), 
                   font, 0.6, (0, 0, 255), 2)
    
    if smooth_pos_1:
        cv2.putText(dashboard, "PINK: OK", (GRAPH_WIDTH + 180, status_y), 
                   font, 0.6, (255, 0, 255), 2)
    else:
        cv2.putText(dashboard, "PINK: LOST", (GRAPH_WIDTH + 180, status_y), 
                   font, 0.6, (0, 0, 255), 2)

    if recording:
        out.write(dashboard)
        cv2.circle(dashboard, (DASHBOARD_WIDTH-50, 50), 8, (0, 0, 255), -1)
        cv2.putText(dashboard, "REC", (DASHBOARD_WIDTH-90, 55), font, 0.6, (0, 0, 255), 2)

    cv2.imshow("Chaos Dashboard", dashboard)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('l'):
        if not recording:
            fn = f'dashboard_log_{file_counter}.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(fn, fourcc, actual_fps, (DASHBOARD_WIDTH, DASHBOARD_HEIGHT))
            recording = True
            print(f"Recording: {fn}")
        else:
            recording = False
            out.release()
            file_counter += 1
            print("Saved.")

    if key == ord('c'):
        trail_canvas = np.zeros_like(frame)
        phase_canvas = np.zeros((GRAPH_HEIGHT_TOP, GRAPH_WIDTH, 3), dtype=np.uint8)
        draw_phase_grid(phase_canvas)
        time_theta_1.clear()
        time_theta_2.clear()
        prev_theta = None
        prev_theta1 = None
        prev_theta2 = None
        prev_time = None
        omega1_buffer.clear()
        omega2_buffer.clear()
        omega_max_buffer.clear()
        adaptive_scale = 1.0
        pos_buffer_0.clear()
        pos_buffer_1.clear()
        print("Cleared.")

    if key == ord('q'):
        break

if recording and out: 
    out.release()
cap.release()
cv2.destroyAllWindows()
print("\nShutdown complete")
