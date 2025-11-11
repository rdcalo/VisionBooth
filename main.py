import os
import cv2
import datetime
import time
from gesture_detector import GestureDetector

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CONSECUTIVE_REQUIRED = 5
COUNTDOWN_POS = (IMAGE_WIDTH // 2 - 50, IMAGE_HEIGHT // 2)

if not os.path.exists("sessions"):
    os.mkdir("sessions")

SESSION_DIR = f"sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.mkdir(SESSION_DIR)

gesture_detector = GestureDetector()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

def overlay_text(img, text, pos, color, scale=0.8, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def capture_and_save(frame):
    filename = f"{SESSION_DIR}/photo_{datetime.datetime.now().strftime('%H%M%S')}.png"
    cv2.imwrite(filename, frame)
    print(f"[Saved] {filename}")

class State:
    PROMPT_TIMER = "PROMPT_TIMER"
    DETECTING_FINGERS = "DETECTING_FINGERS"
    TIMER_SET = "TIMER_SET"
    AWAIT_THUMBS_UP = "AWAIT_THUMBS_UP"
    COUNTDOWN = "COUNTDOWN"
    CAPTURE_DONE = "CAPTURE_DONE"

state = State.PROMPT_TIMER
last_count = None
count_streak = 0
thumb_up_streak = 0
fist_streak = 0
timer_value = None
countdown_end = None

print("Show your fingers to set the timer (1–5).")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    frame, gesture_name = gesture_detector.detect_gesture(frame)
    current_time = time.time()
    
    finger_count_map = {
        "One Finger": 1,
        "Peace Sign": 2,
        "Three Fingers": 3,
        "Four Fingers": 4,
        "Open Palm": 5
    }
    
    detected_count = finger_count_map.get(gesture_name)
    
    thumb_up = (gesture_name == "Thumbs Up")
    fist_detected = (gesture_name == "Fist")

    if state == State.PROMPT_TIMER:
        overlay_text(frame, "Welcome to PhotoBooth!", (10, 40), (0, 255, 255), 1.2, 3)
        overlay_text(frame, "Show 1-5 fingers to set your timer", (10, 90), (255, 255, 255), 0.9, 2)
        
        if detected_count and 1 <= detected_count <= 5:
            state = State.DETECTING_FINGERS
            last_count = detected_count
            count_streak = 1

    elif state == State.DETECTING_FINGERS:
        overlay_text(frame, "Welcome to PhotoBooth!", (10, 40), (0, 255, 255), 1.2, 3)
        overlay_text(frame, "Show 1-5 fingers to set your timer", (10, 90), (255, 255, 255), 0.9, 2)
        
        if detected_count:
            if detected_count == last_count and 1 <= detected_count <= 5:
                count_streak += 1
                # Show progress indicator
                progress = "●" * count_streak + "○" * (CONSECUTIVE_REQUIRED - count_streak)
                overlay_text(frame, f"Detecting {detected_count} fingers... {progress}", 
                           (10, 140), (0, 255, 0), 0.8, 2)
                
                if count_streak >= CONSECUTIVE_REQUIRED:
                    timer_value = detected_count
                    print(f"Timer set to: {timer_value}s")
                    state = State.TIMER_SET
                    count_streak = 0
                    thumb_up_streak = 0
            else:
                count_streak = 1
                last_count = detected_count
        else:
            # No hand detected, go back to prompt
            state = State.PROMPT_TIMER
            last_count = None
            count_streak = 0

    elif state == State.TIMER_SET:
        overlay_text(frame, f"Timer set to {timer_value} seconds!", (10, 40), (0, 255, 0), 1.2, 3)
        overlay_text(frame, "Thumbs up to START | Fist to CHANGE", (10, 90), (255, 255, 0), 0.9, 2)
        state = State.AWAIT_THUMBS_UP

    elif state == State.AWAIT_THUMBS_UP:
        overlay_text(frame, f"Timer set to {timer_value} seconds!", (10, 40), (0, 255, 0), 1.2, 3)
        overlay_text(frame, "Thumbs up to START | Fist to CHANGE", (10, 90), (255, 255, 0), 0.9, 2)

        if thumb_up:
            thumb_up_streak += 1
            fist_streak = 0
            progress = "●" * thumb_up_streak + "○" * (CONSECUTIVE_REQUIRED - thumb_up_streak)
            overlay_text(frame, f"Starting... {progress}", (10, 140), (0, 255, 255), 0.8, 2)
            
            if thumb_up_streak >= CONSECUTIVE_REQUIRED:
                countdown_end = current_time + timer_value
                print(f"Starting countdown: {timer_value}s")
                state = State.COUNTDOWN
                fist_streak = 0
        elif fist_detected:
            fist_streak += 1
            thumb_up_streak = 0
            progress = "●" * fist_streak + "○" * (CONSECUTIVE_REQUIRED - fist_streak)
            overlay_text(frame, f"Resetting timer... {progress}", (10, 140), (255, 150, 0), 0.8, 2)
            
            if fist_streak >= CONSECUTIVE_REQUIRED:
                print(f"Resetting timer, back to finger selection")
                state = State.PROMPT_TIMER
                timer_value = None
                last_count = None
                count_streak = 0
                thumb_up_streak = 0
                fist_streak = 0
        else:
            thumb_up_streak = 0
            fist_streak = 0


    elif state == State.COUNTDOWN:
        remaining = max(0, int(round(countdown_end - current_time)))
        
        if remaining > 0:
            overlay_text(frame, str(remaining), COUNTDOWN_POS, (0, 0, 255), 3.5, 10)
        else:
            overlay_text(frame, "Say Cheese!", COUNTDOWN_POS, (0, 255, 0), 2.0, 6)
        
        if current_time >= countdown_end:
            capture_and_save(frame)
            state = State.CAPTURE_DONE


    elif state == State.CAPTURE_DONE:
        overlay_text(frame, "Photo captured!", (10, 40), (255, 255, 0), 1.2, 3)
        overlay_text(frame, "Show fingers to take another photo", (10, 90), (255, 255, 255), 0.9, 2)

        time.sleep(1.5)
        state = State.PROMPT_TIMER
        timer_value = None
        last_count = None
        count_streak = 0
        thumb_up_streak = 0

    cv2.imshow("PhotoBooth", frame)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
print("Exiting PhotoBooth")