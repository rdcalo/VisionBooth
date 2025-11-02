from flask import Flask, render_template, Response, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from gesture_detector import GestureDetector
import datetime
import os
import time
from threading import Lock
import logging
from PIL import Image, ImageDraw, ImageFont
import io

# Reduce Flask logging for cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'photobooth_secret'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    max_http_buffer_size=50 * 1024 * 1024,  # Increase buffer for large images
    ping_timeout=60,  # Increase ping timeout
    ping_interval=25  # Increase ping interval
)

# Initialize gesture detector with optimized settings
gesture_detector = GestureDetector()

# Optimize MediaPipe settings if possible
try:
    gesture_detector.hands.min_detection_confidence = 0.6
    gesture_detector.hands.min_tracking_confidence = 0.5
except:
    pass

# Session management
if not os.path.exists("sessions"):
    os.mkdir("sessions")

SESSION_DIR = None
current_state = {
    'state': 'PROMPT_TIMER',
    'timer_value': None,
    'countdown_end': None,
    'detected_gesture': None,
    'last_count': None,
    'count_streak': 0,
    'thumb_up_streak': 0,
    'fist_streak': 0,
    'capture_count': 0,  # NEW: Track captures (0-4)
    'captured_images': [],  # NEW: Store base64 images
    'strip_filename': None  # NEW: Final strip path
}
state_lock = Lock()

CONSECUTIVE_REQUIRED = 5
PHOTOS_PER_STRIP = 4

# ==============================
# Routes
# ==============================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sessions/<path:filename>')
def serve_photo(filename):
    return send_from_directory('sessions', filename)

# ==============================
# WebSocket Events
# ==============================

@socketio.on('connect')
def handle_connect():
    global SESSION_DIR
    SESSION_DIR = f"sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(SESSION_DIR, exist_ok=True)
    print(f"✅ Client connected. Session: {SESSION_DIR}")
    emit('connected', {'session': SESSION_DIR})

@socketio.on('disconnect')
def handle_disconnect():
    print("❌ Client disconnected")

@socketio.on('video_frame')
def handle_video_frame(data):
    global current_state
    
    try:
        # Decode base64 image - OPTIMIZED
        img_str = data['image'].split(',')[1]
        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None or frame.size == 0:
            emit('state_update', {
                'frame': data['image'],
                'state': current_state['state'],
                'timer_value': current_state['timer_value'],
                'gesture': None,
                'countdown': get_countdown(),
                'streak_progress': get_streak_progress(),
                'capture_count': current_state['capture_count'],
                'total_captures': PHOTOS_PER_STRIP
            })
            return
        
        # Detect gesture - wrapped in try-catch for MediaPipe errors
        try:
            frame, gesture_name = gesture_detector.detect_gesture(frame)
        except Exception as gesture_error:
            # If gesture detection fails, continue with None gesture
            print(f"⚠️ Gesture detection error: {gesture_error}")
            gesture_name = None
        
        with state_lock:
            current_state['detected_gesture'] = gesture_name
            process_state_machine(gesture_name)
            
            # Encode frame back to base64 with aggressive JPEG compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            success, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if not success:
                return
                
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send state and frame back to client
            emit('state_update', {
                'frame': f'data:image/jpeg;base64,{frame_base64}',
                'state': current_state['state'],
                'timer_value': current_state['timer_value'],
                'gesture': gesture_name,
                'countdown': get_countdown(),
                'streak_progress': get_streak_progress(),
                'trigger_capture': (current_state['state'] == 'CAPTURE_DONE'),
                'capture_count': current_state['capture_count'],
                'total_captures': PHOTOS_PER_STRIP,
                'strip_ready': current_state['capture_count'] >= PHOTOS_PER_STRIP,
                'strip_filename': current_state['strip_filename']
            })
    
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        emit('state_update', {
            'frame': data.get('image', ''),
            'state': current_state['state'],
            'timer_value': current_state['timer_value'],
            'gesture': None,
            'countdown': get_countdown(),
            'streak_progress': get_streak_progress(),
            'trigger_capture': False,
            'capture_count': current_state['capture_count'],
            'total_captures': PHOTOS_PER_STRIP
        })

def process_state_machine(gesture_name):
    """Process the photobooth state machine"""
    global current_state
    
    state = current_state['state']
    current_time = time.time()
    
    # Map gestures to finger counts
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
    
    # STATE: PROMPT_TIMER
    if state == 'PROMPT_TIMER':
        if detected_count and 1 <= detected_count <= 5:
            current_state['state'] = 'DETECTING_FINGERS'
            current_state['last_count'] = detected_count
            current_state['count_streak'] = 1
    
    # STATE: DETECTING_FINGERS
    elif state == 'DETECTING_FINGERS':
        if detected_count:
            if detected_count == current_state['last_count'] and 1 <= detected_count <= 5:
                current_state['count_streak'] += 1
                if current_state['count_streak'] >= CONSECUTIVE_REQUIRED:
                    current_state['timer_value'] = detected_count
                    current_state['state'] = 'TIMER_SET'
                    current_state['count_streak'] = 0
                    print(f"⏱ Timer set to: {detected_count}s")
            else:
                current_state['count_streak'] = 1
                current_state['last_count'] = detected_count
        else:
            current_state['state'] = 'PROMPT_TIMER'
            current_state['last_count'] = None
            current_state['count_streak'] = 0
    
    # STATE: TIMER_SET
    elif state == 'TIMER_SET':
        current_state['state'] = 'AWAIT_THUMBS_UP'
        current_state['thumb_up_streak'] = 0
        current_state['fist_streak'] = 0
    
    # STATE: AWAIT_THUMBS_UP
    elif state == 'AWAIT_THUMBS_UP':
        if thumb_up:
            current_state['thumb_up_streak'] += 1
            current_state['fist_streak'] = 0
            if current_state['thumb_up_streak'] >= CONSECUTIVE_REQUIRED:
                current_state['countdown_end'] = current_time + current_state['timer_value']
                current_state['state'] = 'COUNTDOWN'
                print(f"▶ Starting countdown: {current_state['timer_value']}s")
        elif fist_detected:
            current_state['fist_streak'] += 1
            current_state['thumb_up_streak'] = 0
            if current_state['fist_streak'] >= CONSECUTIVE_REQUIRED:
                print("🔄 Resetting timer")
                reset_to_prompt()
        else:
            current_state['thumb_up_streak'] = 0
            current_state['fist_streak'] = 0
    
    # STATE: COUNTDOWN
    elif state == 'COUNTDOWN':
        remaining = get_countdown()
        if remaining is not None and remaining <= 0:
            current_state['state'] = 'CAPTURE_DONE'
            print(f"📸 Triggering photo capture {current_state['capture_count'] + 1}/{PHOTOS_PER_STRIP}...")
    
    # STATE: CAPTURE_DONE
    elif state == 'CAPTURE_DONE':
        # Wait for photo to be saved via save_photo handler
        # Don't do anything here - let save_photo handle the next countdown
        pass

def reset_to_prompt():
    """Reset state machine to initial state"""
    global current_state
    current_state = {
        'state': 'PROMPT_TIMER',
        'timer_value': None,
        'countdown_end': None,
        'detected_gesture': None,
        'last_count': None,
        'count_streak': 0,
        'thumb_up_streak': 0,
        'fist_streak': 0,
        'capture_count': 0,
        'captured_images': [],
        'strip_filename': None
    }

def get_countdown():
    """Get remaining countdown time"""
    if current_state['state'] == 'COUNTDOWN' and current_state['countdown_end']:
        remaining = current_state['countdown_end'] - time.time()
        return max(0, int(round(remaining)))
    return None

def get_streak_progress():
    """Get progress for gesture streaks"""
    if current_state['state'] == 'DETECTING_FINGERS':
        return {'current': current_state['count_streak'], 'required': CONSECUTIVE_REQUIRED}
    elif current_state['state'] == 'AWAIT_THUMBS_UP':
        if current_state['thumb_up_streak'] > 0:
            return {'current': current_state['thumb_up_streak'], 'required': CONSECUTIVE_REQUIRED}
        elif current_state['fist_streak'] > 0:
            return {'current': current_state['fist_streak'], 'required': CONSECUTIVE_REQUIRED}
    return None

@socketio.on('save_photo')
def handle_save_photo(data):
    """Save captured photo and continue sequence"""
    global SESSION_DIR
    
    print(f"📸 Received photo save request. Current count: {current_state['capture_count']}")
    
    try:
        # Prevent duplicate captures beyond limit
        if current_state['capture_count'] >= PHOTOS_PER_STRIP:
            print(f"⚠️ Already captured {current_state['capture_count']} photos, ignoring duplicate")
            return
        
        # Ensure session directory exists
        if SESSION_DIR is None or not os.path.exists(SESSION_DIR):
            SESSION_DIR = f"sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(SESSION_DIR, exist_ok=True)
            print(f"📁 Created session directory: {SESSION_DIR}")
        
        # Validate image data
        if 'image' not in data or not data['image']:
            print("❌ No image data received")
            emit('photo_error', {'error': 'No image data'})
            return
        
        # Store the image data for strip generation
        img_data = data['image']
        print(f"📦 Image data size: {len(img_data) / 1024:.2f} KB")
        
        current_state['captured_images'].append(img_data)
        current_state['capture_count'] += 1
        
        print(f"✅ Photo {current_state['capture_count']}/{PHOTOS_PER_STRIP} captured and stored")
        
        # Emit acknowledgment immediately
        emit('photo_received', {
            'count': current_state['capture_count'],
            'total': PHOTOS_PER_STRIP
        })
        
        # Check if we've captured all photos
        if current_state['capture_count'] >= PHOTOS_PER_STRIP:
            print("🎨 Generating photo strip...")
            
            # Change state to prevent more captures
            current_state['state'] = 'STRIP_GENERATING'
            
            # Generate photo strip
            strip_filename = create_photo_strip(current_state['captured_images'], SESSION_DIR)
            
            if strip_filename:
                current_state['strip_filename'] = strip_filename
                print(f"🎉 Photo strip complete! Saved as: {strip_filename}")
                
                # Send strip ready notification
                emit('strip_ready', {
                    'filename': strip_filename,
                    'message': 'Photo strip is ready!'
                })
            else:
                print("❌ Failed to create photo strip")
                emit('photo_error', {'error': 'Failed to create strip'})
            
            # Reset for next session after a delay
            time.sleep(1)
            reset_to_prompt()
        else:
            # Validate timer_value before continuing
            if current_state['timer_value'] is None or current_state['timer_value'] <= 0:
                print("❌ Invalid timer_value, resetting")
                reset_to_prompt()
                return
            
            # NOW restart countdown for next photo (after current photo is saved)
            print(f"⏭ Preparing for photo {current_state['capture_count'] + 1}/{PHOTOS_PER_STRIP}")
            
            # Brief pause before starting next countdown
            time.sleep(1)
            
            current_time = time.time()
            current_state['countdown_end'] = current_time + current_state['timer_value']
            current_state['state'] = 'COUNTDOWN'
            
            print(f"▶ Starting countdown for photo {current_state['capture_count'] + 1}/{PHOTOS_PER_STRIP}")
        
    except Exception as e:
        print(f"❌ Error saving photo: {e}")
        import traceback
        traceback.print_exc()
        emit('photo_error', {'error': str(e)})
        # Reset on error to prevent stuck state
        reset_to_prompt()

def create_photo_strip(images, session_dir):
    """Create a vertical photo strip from captured images"""
    try:
        # Configuration
        PHOTO_WIDTH = 800
        PHOTO_HEIGHT = 600
        BORDER_SIZE = 20
        FRAME_COLOR = (173, 216, 230)  # Light blue frame
        
        # Calculate strip dimensions
        strip_width = PHOTO_WIDTH + (BORDER_SIZE * 2)
        strip_height = (PHOTO_HEIGHT * PHOTOS_PER_STRIP) + (BORDER_SIZE * (PHOTOS_PER_STRIP + 1))
        
        # Create blank canvas with border color
        strip = Image.new('RGB', (strip_width, strip_height), FRAME_COLOR)
        
        # Paste each photo
        for idx, img_data in enumerate(images):
            # Decode base64 image
            img_str = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_str)
            
            # Open with PIL
            photo = Image.open(io.BytesIO(img_bytes))
            
            # Resize to standard size
            photo = photo.resize((PHOTO_WIDTH, PHOTO_HEIGHT), Image.Resampling.LANCZOS)
            
            # Calculate position
            y_position = BORDER_SIZE + (idx * (PHOTO_HEIGHT + BORDER_SIZE))
            
            # Paste photo onto strip
            strip.paste(photo, (BORDER_SIZE, y_position))
        
        # Add timestamp at bottom
        draw = ImageDraw.Draw(strip)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Try to use a nice font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Draw text at bottom center
        text_bbox = draw.textbbox((0, 0), timestamp, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (strip_width - text_width) // 2
        text_y = strip_height - BORDER_SIZE + 5
        
        # Add text shadow for visibility
        draw.text((text_x + 2, text_y + 2), timestamp, fill=(0, 0, 0), font=font)
        draw.text((text_x, text_y), timestamp, fill=(255, 255, 255), font=font)
        
        # Save the strip
        timestamp_filename = datetime.datetime.now().strftime('%H%M%S')
        filename = f"strip_{timestamp_filename}.png"
        full_path = os.path.join(session_dir, filename)
        
        strip.save(full_path, 'PNG')
        
        # Verify and return relative path
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            print(f"✅ [Strip Saved] {full_path} ({file_size} bytes)")
            return f"{session_dir}/{filename}"
        else:
            print(f"❌ Failed to save strip: {full_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating photo strip: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==============================
# Run App
# ==============================

if __name__ == '__main__':
    print("=" * 50)
    print("🎉 PhotoBooth Web App Starting...")
    print("📸 Open your browser to: http://localhost:5000")
    print("=" * 50)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)