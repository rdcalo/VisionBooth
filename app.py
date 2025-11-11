import os
os.environ['GLOG_minloglevel'] = '3'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

from flask import Flask, render_template, Response, jsonify, send_from_directory, url_for
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from gesture_detector import GestureDetector
import datetime
import time
from threading import Lock
import logging
from PIL import Image, ImageDraw, ImageFont
import io


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'photobooth_secret'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    max_http_buffer_size=50 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25
)


gesture_detector = GestureDetector()
try:
    gesture_detector.hands.min_detection_confidence = 0.6
    gesture_detector.hands.min_tracking_confidence = 0.5
except:
    pass


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
    'capture_count': 0,
    'captured_images': [],
    'strip_filename': None
}
state_lock = Lock()

CONSECUTIVE_REQUIRED = 5
PHOTOS_PER_STRIP = 4


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/index')
def photobooth():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/sessions/<path:filename>')
def serve_photo(filename):
    return send_from_directory('sessions', filename)


@socketio.on('connect')
def handle_connect():
    global SESSION_DIR
    SESSION_DIR = f"sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(SESSION_DIR, exist_ok=True)
    print(f"Client connected. Session: {SESSION_DIR}")
    emit('connected', {'session': SESSION_DIR})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('video_frame')
def handle_video_frame(data):
    global current_state

    try:

        img_str = data['image'].split(',')[1]
        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            emit('state_update', get_default_state(data['image']))
            return

        try:
            frame, gesture_name = gesture_detector.detect_gesture(frame)
        except Exception as gesture_error:
            print(f"Gesture detection error: {gesture_error}")
            gesture_name = None

        with state_lock:
            current_state['detected_gesture'] = gesture_name
            process_state_machine(gesture_name)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            success, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not success:
                return

            frame_base64 = base64.b64encode(buffer).decode('utf-8')

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
        print(f"Error processing frame: {e}")
        emit('state_update', get_default_state(data.get('image', '')))

def get_default_state(image):
    return {
        'frame': image,
        'state': current_state['state'],
        'timer_value': current_state['timer_value'],
        'gesture': None,
        'countdown': get_countdown(),
        'streak_progress': get_streak_progress(),
        'trigger_capture': False,
        'capture_count': current_state['capture_count'],
        'total_captures': PHOTOS_PER_STRIP
    }


def process_state_machine(gesture_name):
    global current_state

    state = current_state['state']
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

    if state == 'PROMPT_TIMER':
        if detected_count and 1 <= detected_count <= 5:
            current_state.update({'state': 'DETECTING_FINGERS', 'last_count': detected_count, 'count_streak': 1})

    elif state == 'DETECTING_FINGERS':
        if detected_count:
            if detected_count == current_state['last_count']:
                current_state['count_streak'] += 1
                if current_state['count_streak'] >= CONSECUTIVE_REQUIRED:
                    current_state.update({'timer_value': detected_count, 'state': 'TIMER_SET', 'count_streak': 0})
                    print(f"Timer set to: {detected_count}s")
            else:
                current_state.update({'count_streak': 1, 'last_count': detected_count})
        else:
            reset_to_prompt()

    elif state == 'TIMER_SET':
        current_state.update({'state': 'AWAIT_THUMBS_UP', 'thumb_up_streak': 0, 'fist_streak': 0})

    elif state == 'AWAIT_THUMBS_UP':
        if thumb_up:
            current_state['thumb_up_streak'] += 1
            if current_state['thumb_up_streak'] >= CONSECUTIVE_REQUIRED:
                current_state.update({'countdown_end': current_time + current_state['timer_value'], 'state': 'COUNTDOWN'})
                print(f"▶ Starting countdown: {current_state['timer_value']}s")
        elif fist_detected:
            current_state['fist_streak'] += 1
            if current_state['fist_streak'] >= CONSECUTIVE_REQUIRED:
                print("Resetting timer")
                reset_to_prompt()
        else:
            current_state['thumb_up_streak'] = current_state['fist_streak'] = 0

    elif state == 'COUNTDOWN':
        if get_countdown() is not None and get_countdown() <= 0:
            current_state.update({'state': 'CAPTURE_DONE', 'countdown_end': None})
            print(f"Capture {current_state['capture_count'] + 1}/{PHOTOS_PER_STRIP}")

def reset_to_prompt():
    global current_state
    current_state.update({
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
    })

def get_countdown():
    if current_state['state'] == 'COUNTDOWN' and current_state['countdown_end']:
        remaining = current_state['countdown_end'] - time.time()
        return max(0, int(round(remaining)))
    return None

def get_streak_progress():
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
    global SESSION_DIR

    try:
        if current_state['capture_count'] >= PHOTOS_PER_STRIP:
            return

        if SESSION_DIR is None or not os.path.exists(SESSION_DIR):
            SESSION_DIR = f"sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(SESSION_DIR, exist_ok=True)

        img_data = data.get('image')
        if not img_data:
            emit('photo_error', {'error': 'No image data'})
            return

        current_state['captured_images'].append(img_data)
        current_state['capture_count'] += 1

        emit('photo_received', {'count': current_state['capture_count'], 'total': PHOTOS_PER_STRIP})

        if current_state['capture_count'] >= PHOTOS_PER_STRIP:
            current_state['state'] = 'STRIP_GENERATING'
            strip_filename = create_photo_strip(current_state['captured_images'], SESSION_DIR)
            if strip_filename:
                current_state['strip_filename'] = strip_filename
                emit('strip_ready', {'filename': strip_filename, 'message': 'Photo strip ready!'})
            reset_to_prompt()
        else:
            time.sleep(1)
            current_state.update({
                'countdown_end': time.time() + current_state['timer_value'],
                'state': 'COUNTDOWN'
            })

    except Exception as e:
        print(f"Error saving photo: {e}")
        reset_to_prompt()
        emit('photo_error', {'error': str(e)})



def create_photo_strip(images, session_dir):
    try:
       
        DPI = 300
        STRIP_WIDTH_MM = 51
        STRIP_HEIGHT_MM = 152
        
    
        MM_TO_INCH = 0.0393701
        STRIP_WIDTH_PX = int(STRIP_WIDTH_MM * MM_TO_INCH * DPI)  # ~602px
        STRIP_HEIGHT_PX = int(STRIP_HEIGHT_MM * MM_TO_INCH * DPI)  # ~1795px
        
 
        TOP_BORDER = 60
        SIDE_BORDER = 60
        PHOTO_SPACING = 40
        BOTTOM_AREA = 100  
        
        available_height = STRIP_HEIGHT_PX - TOP_BORDER - BOTTOM_AREA - (PHOTO_SPACING * (PHOTOS_PER_STRIP - 1))
        PHOTO_HEIGHT = available_height // PHOTOS_PER_STRIP
        PHOTO_WIDTH = STRIP_WIDTH_PX - (SIDE_BORDER * 2)
        
        print(f"Strip size: {STRIP_WIDTH_PX}x{STRIP_HEIGHT_PX}px ({STRIP_WIDTH_MM}x{STRIP_HEIGHT_MM}mm)")
        print(f"Photo slots: {PHOTO_WIDTH}x{PHOTO_HEIGHT}px")
        
        FRAME_COLOR = (255, 255, 255)
        strip = Image.new('RGB', (STRIP_WIDTH_PX, STRIP_HEIGHT_PX), FRAME_COLOR)

        for i, img_data in enumerate(images):
            img_bytes = base64.b64decode(img_data.split(',')[1])
            photo = Image.open(io.BytesIO(img_bytes))
            
    
            target_aspect = PHOTO_WIDTH / PHOTO_HEIGHT
            photo_aspect = photo.width / photo.height

            if photo_aspect > target_aspect:
            
                new_width = int(photo.height * target_aspect)
                left = (photo.width - new_width) // 2
                photo = photo.crop((left, 0, left + new_width, photo.height))
            else:
          
                new_height = int(photo.width / target_aspect)
                top = (photo.height - new_height) // 2
                photo = photo.crop((0, top, photo.width, top + new_height))
            
            photo = photo.resize((PHOTO_WIDTH, PHOTO_HEIGHT), Image.Resampling.LANCZOS)

            y_pos = TOP_BORDER + i * (PHOTO_HEIGHT + PHOTO_SPACING)
            strip.paste(photo, (SIDE_BORDER, y_pos))
        
        draw = ImageDraw.Draw(strip)

        now = datetime.datetime.now()
        branding_text = f"VisionBooth {now.strftime('%m/%d/%y')}"

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 36)
            except:
                font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), branding_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = (STRIP_WIDTH_PX - text_width) // 2
        text_y = STRIP_HEIGHT_PX - BOTTOM_AREA + (BOTTOM_AREA - text_height) // 2

        shadow_offset = 2
        draw.text((text_x + shadow_offset, text_y + shadow_offset), branding_text, fill=(200, 200, 200), font=font)
        draw.text((text_x, text_y), branding_text, fill=(50, 50, 50), font=font)

        filename = f"strip_{now.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(session_dir, filename)
        strip.save(path, dpi=(DPI, DPI), quality=95, optimize=False)
        
        print(f"Strip saved: {path}")
        print(f"   Size: {STRIP_WIDTH_PX}x{STRIP_HEIGHT_PX}px | {STRIP_WIDTH_MM}x{STRIP_HEIGHT_MM}mm @ {DPI}DPI")
        
        return f"{session_dir}/{filename}" if os.path.exists(path) else None
        
    except Exception as e:
        print(f"❌ Error creating strip: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("=" * 50)
    print("VisionBooth Starting...")
    print("Open browser at: http://localhost:5000")
    print("=" * 50)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
