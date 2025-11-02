import cv2
import mediapipe as mp
import math
import time

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Add timestamp tracking to prevent MediaPipe errors
        self.last_timestamp = 0
        self.frame_counter = 0

    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def detect_gesture(self, frame):
        """Detect gesture with error handling for MediaPipe timestamp issues"""
        
        # Validate frame first
        if frame is None or frame.size == 0:
            return frame, None
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"⚠️ Frame conversion error: {e}")
            return frame, None
        
        # Process frame with MediaPipe - wrapped in try-catch for timestamp errors
        gesture_name = None
        try:
            results = self.hands.process(frame_rgb)
        except Exception as e:
            # Catch MediaPipe timestamp errors and continue
            if "timestamp" in str(e).lower():
                print(f"⚠️ MediaPipe timestamp error (ignoring): {e}")
            else:
                print(f"❌ MediaPipe error: {e}")
            return frame, None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            handedness = results.multi_handedness[0].classification[0].label
            
            # Get key landmarks
            wrist = landmarks[0]
            
            # Thumb landmarks
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            thumb_mcp = landmarks[2]
            thumb_cmc = landmarks[1]
            
            # Other finger landmarks
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            index_mcp = landmarks[5]
            
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            middle_mcp = landmarks[9]
            
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            ring_mcp = landmarks[13]
            
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            pinky_mcp = landmarks[17]
            
            # === IMPROVED FINGER DETECTION ===
            # Use relative distances (compared to hand size) to handle varying distances from camera
            
            # Calculate hand size (distance from wrist to middle finger MCP as reference)
            hand_size = self.distance(wrist, middle_mcp)
            
            # Calculate distances from fingertips to their MCP (knuckle) joints
            index_dist = self.distance(index_tip, index_mcp)
            middle_dist = self.distance(middle_tip, middle_mcp)
            ring_dist = self.distance(ring_tip, ring_mcp)
            pinky_dist = self.distance(pinky_tip, pinky_mcp)
            thumb_dist = self.distance(thumb_tip, thumb_cmc)
            
            # Normalize distances by hand size
            index_ratio = index_dist / hand_size if hand_size > 0 else 0
            middle_ratio = middle_dist / hand_size if hand_size > 0 else 0
            ring_ratio = ring_dist / hand_size if hand_size > 0 else 0
            pinky_ratio = pinky_dist / hand_size if hand_size > 0 else 0
            thumb_ratio = thumb_dist / hand_size if hand_size > 0 else 0
            
            # Check if fingers are extended using ratios (works at any distance!)
            FINGER_RATIO_THRESHOLD = 0.6
            index_extended = (index_pip.y - index_tip.y) > 0.02 and index_ratio > FINGER_RATIO_THRESHOLD
            middle_extended = (middle_pip.y - middle_tip.y) > 0.02 and middle_ratio > FINGER_RATIO_THRESHOLD
            ring_extended = (ring_pip.y - ring_tip.y) > 0.02 and ring_ratio > FINGER_RATIO_THRESHOLD
            pinky_extended = (pinky_pip.y - pinky_tip.y) > 0.02 and pinky_ratio > FINGER_RATIO_THRESHOLD
            
            # Thumb: Check distance from thumb to index MCP (normalized)
            thumb_to_index_dist = self.distance(thumb_tip, index_mcp)
            thumb_ratio_to_index = thumb_to_index_dist / hand_size if hand_size > 0 else 0
            thumb_extended = thumb_ratio_to_index > 0.7  # Lowered from 0.8
            
            # === SPECIAL GESTURE CHECKS ===
            
            # THUMBS UP: Thumb pointing upward
            # - Thumb extended (far from palm)
            # - Thumb tip is above thumb MCP (pointing up)
            # - All other fingers curled
            thumb_pointing_up = (thumb_mcp.y - thumb_tip.y) > 0.08
            is_thumbs_up = (
                thumb_extended and
                thumb_pointing_up and
                not index_extended and
                not middle_extended and
                not ring_extended and
                not pinky_extended
            )
            
            # FIST: All fingers curled, including thumb
            # - All fingertips close to their MCPs
            # - Thumb close to index finger
            is_fist = (
                not index_extended and
                not middle_extended and
                not ring_extended and
                not pinky_extended and
                thumb_ratio_to_index < 0.7  # Thumb tucked in (relative)
            )
            
            # Count extended fingers (excluding thumb for most gestures)
            fingers_up = [index_extended, middle_extended, ring_extended, pinky_extended]
            finger_count = sum(fingers_up)
            
            # === GESTURE PRIORITY ===
            
            # 1. FIST - Must be checked first
            if is_fist:
                gesture_name = "Fist"
            
            # 2. THUMBS UP - Must be checked before one finger
            elif is_thumbs_up:
                gesture_name = "Thumbs Up"
            
            # 3. ONE FINGER - Only index extended
            elif (index_extended and 
                  not middle_extended and 
                  not ring_extended and 
                  not pinky_extended):
                gesture_name = "One Finger"
            
            # 4. PEACE SIGN - Index and middle
            elif (index_extended and 
                  middle_extended and 
                  not ring_extended and 
                  not pinky_extended):
                gesture_name = "Peace Sign"
            
            # 5. THREE FINGERS
            elif (index_extended and 
                  middle_extended and 
                  ring_extended and 
                  not pinky_extended):
                gesture_name = "Three Fingers"
            
            # 6. FOUR FINGERS
            elif finger_count == 4 and not thumb_extended:
                gesture_name = "Four Fingers"
            
            # 7. OPEN PALM - All five (be lenient with thumb detection)
            elif finger_count == 4 and thumb_extended:
                gesture_name = "Open Palm"
            elif finger_count == 4 and thumb_ratio_to_index > 0.7:  # Fallback with lower threshold
                gesture_name = "Open Palm"

            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )
            
            # Debug: Show detected gesture
            if gesture_name:
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 650), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Gesture: None", (10, 650), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Debug: Show finger states and count
            debug_text = f"Fingers: I:{int(index_extended)} M:{int(middle_extended)} R:{int(ring_extended)} P:{int(pinky_extended)} T:{int(thumb_extended)} = {finger_count+int(thumb_extended)}"
            cv2.putText(frame, debug_text, (10, 680), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Debug: Show ratios (normalized distances)
            debug_dist = f"Ratios: I:{index_ratio:.2f} M:{middle_ratio:.2f} R:{ring_ratio:.2f} P:{pinky_ratio:.2f}"
            cv2.putText(frame, debug_dist, (10, 710), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
            
            # Debug: Show thumb info
            debug_thumb = f"Thumb: Ratio={thumb_ratio_to_index:.2f} Up={thumb_pointing_up} Ext={thumb_extended}"
            cv2.putText(frame, debug_thumb, (10, 735), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)

        return frame, gesture_name