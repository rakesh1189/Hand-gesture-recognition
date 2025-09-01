import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Useful landmark indices (MediaPipe Hands)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

def normalized_to_pixel(coord, width, height):
    return int(coord.x * width), int(coord.y * height)

def finger_is_extended(hand_landmarks, finger_tip_idx, finger_pip_idx, handedness_label):
    
    tip = hand_landmarks.landmark[finger_tip_idx]
    pip = hand_landmarks.landmark[finger_pip_idx]
    # For webcam coordinate system (y increases downward), an extended finger pointing upward has lower y.
    return tip.y < pip.y - 0.02  # small margin

def thumb_is_up(hand_landmarks, handedness_label):
    tip = hand_landmarks.landmark[THUMB_TIP]
    ip = hand_landmarks.landmark[THUMB_IP]
    mcp = hand_landmarks.landmark[THUMB_MCP]
    wrist = hand_landmarks.landmark[WRIST]
    # thumb pointing up: tip.y < mcp.y (higher in image)
    thumb_up_vertical = tip.y < mcp.y - 0.02
    # thumb horizontal position relative to wrist can help determine it's a thumbs-up (not thumbs-left)
    # For right hand, thumb x is to the left or right depending on camera mirroring; we won't rely solely on this.
    return thumb_up_vertical

def classify_gesture(hand_landmarks, handedness_label):
    # Extended state for index, middle, ring, pinky
    idx_ext = finger_is_extended(hand_landmarks, INDEX_TIP, INDEX_PIP, handedness_label)
    mid_ext = finger_is_extended(hand_landmarks, MIDDLE_TIP, MIDDLE_PIP, handedness_label)
    ring_ext = finger_is_extended(hand_landmarks, RING_TIP, RING_PIP, handedness_label)
    pinky_ext = finger_is_extended(hand_landmarks, PINKY_TIP, PINKY_PIP, handedness_label)
    # Thumb check
    thumb_ext = False
    # Heuristic: thumb extended if tip.x is sufficiently away from thumb MCP in the expected horizontal direction
    thumb_tip = hand_landmarks.landmark[THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[THUMB_MCP]
    # Use vector magnitude from thumb MCP to tip
    dx = abs(thumb_tip.x - thumb_mcp.x)
    dy = abs(thumb_tip.y - thumb_mcp.y)
    thumb_ext = dx > 0.06 or dy > 0.06

    # Count extended fingers ignoring thumb for some gestures
    extended_count = sum([idx_ext, mid_ext, ring_ext, pinky_ext]) + (1 if thumb_ext else 0)

    # Rule-based classification (priority order)
    # 1) Fist: no fingers extended (all folded)
    if not idx_ext and not mid_ext and not ring_ext and not pinky_ext and not thumb_ext:
        return "Fist"

    # 2) Open Palm: all main fingers extended and thumb roughly extended
    if idx_ext and mid_ext and ring_ext and pinky_ext and thumb_ext:
        return "Open Palm"

    # 3) Peace Sign: index & middle extended, ring & pinky folded (thumb can be either folded or extended)
    if idx_ext and mid_ext and not ring_ext and not pinky_ext:
        return "Peace (V)"

    # 4) Thumbs Up: thumb extended/up and other fingers folded
    # additional check: thumb direction roughly upwards
    if thumb_is_up(hand_landmarks, handedness_label) and not idx_ext and not mid_ext and not ring_ext and not pinky_ext:
        return "Thumbs Up"

    # If nothing matched, mark unknown but show counts
    return "Unknown"

def draw_label(image, text, origin=(10, 30), bgcolor=(0,0,0), fgcolor=(255,255,255)):
    cv2.rectangle(image, (origin[0]-5, origin[1]-20), (origin[0]+200, origin[1]+10), bgcolor, -1)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, fgcolor, 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {width}x{height}")

    writer = None
    recording = False

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5) as hands:

        prev_time = 0
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty frame.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            label = "No hand"
            if results.multi_hand_landmarks:
                # If multiple hands, we take the first
                hand_landmarks = results.multi_hand_landmarks[0]
                # attempt to get handedness if provided
                handedness_label = None
                if results.multi_handedness:
                    handedness_label = results.multi_handedness[0].classification[0].label

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Classify
                label = classify_gesture(hand_landmarks, handedness_label)

                # Draw bounding box around hand using landmark extremes
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x_min = int(min(xs) * width) - 20
                x_max = int(max(xs) * width) + 20
                y_min = int(min(ys) * height) - 20
                y_max = int(max(ys) * height) + 20
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

            # FPS display
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0.0
            prev_time = curr_time
            fps_text = f"FPS: {int(fps)}"

            draw_label(frame, f"Gesture: {label}", origin=(10,30))
            draw_label(frame, fps_text, origin=(10,60), bgcolor=(50,50,50))

            cv2.imshow("Hand Gesture Recognition", frame)

            # If recording, write frame
            if recording and writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # toggle recording
                if not recording:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter('demo.mp4', fourcc, 20.0, (width, height))
                    recording = True
                    print("Recording started -> demo.mp4")
                else:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    print("Recording stopped")
            elif key == ord('s'):
                cv2.imwrite('snapshot.png', frame)
                print("Saved snapshot.png")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
