from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import sqlite3
import os
import re
import base64
from datetime import datetime
import easyocr
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize EasyOCR reader (used for text extraction from detected regions)
reader = easyocr.Reader(['en'], verbose=False)

# Thread lock for DB operations
db_lock = threading.Lock()

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('bus_tracking.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS bus_in_campus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bus_id TEXT NOT NULL,
            number_plate TEXT NOT NULL,
            bus_number TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            image_snapshot TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS bus_out_campus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bus_id TEXT NOT NULL,
            number_plate TEXT NOT NULL,
            bus_number TEXT NOT NULL,
            exit_time TEXT NOT NULL,
            exit_date TEXT NOT NULL,
            image_snapshot TEXT
        )
    ''')
    # Track current status of each bus
    c.execute('''
        CREATE TABLE IF NOT EXISTS bus_status (
            bus_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            last_seen TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# CNN / DETECTION HELPERS
# ─────────────────────────────────────────────
def preprocess_for_ocr(roi):
    """Enhance ROI for better OCR accuracy."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Resize for better OCR
    h, w = gray.shape
    scale = max(1, 100 // h)
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    # Denoise + threshold
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_number_plate(image):
    """
    Detect number plate region using Haar Cascade + morphological operations.
    Falls back to full-image OCR if no plate region is found.
    Returns (text, confidence, bbox)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try Haar cascade for license plate
    cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 20))

    best_text = ""
    bbox = None

    if len(plates) > 0:
        # Pick largest detected plate
        plates = sorted(plates, key=lambda p: p[2] * p[3], reverse=True)
        x, y, w, h = plates[0]
        # Expand bbox slightly
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        roi = image[y1:y2, x1:x2]
        processed = preprocess_for_ocr(roi)
        results = reader.readtext(processed, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        for (_, text, conf) in results:
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(clean) >= 4:
                best_text = clean
                bbox = (x1, y1, x2, y2)
                break

    if not best_text:
        # Fallback: run OCR on upper-half and lower-half of image
        h, w = image.shape[:2]
        for region in [image[:h//2, :], image[h//2:, :]]:
            processed = preprocess_for_ocr(region)
            results = reader.readtext(processed, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            for (_, text, conf) in results:
                clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean) >= 4:
                    best_text = clean
                    break
            if best_text:
                break

    return best_text, bbox

def detect_bus_number(image):
    """
    Detect bus route number from the FRONT of the bus.
    Looks for 1–3 digit standalone numbers in the upper portion.
    """
    h, w = image.shape[:2]
    # Focus on upper-center area where bus numbers typically appear
    roi = image[:int(h * 0.4), w//4: 3*w//4]
    processed = preprocess_for_ocr(roi)
    results = reader.readtext(processed, detail=1, allowlist='0123456789')
    for (_, text, conf) in results:
        clean = re.sub(r'\D', '', text)
        if 1 <= len(clean) <= 3 and conf > 0.4:
            return clean
    # Wider fallback
    results = reader.readtext(image, detail=1, allowlist='0123456789')
    for (_, text, conf) in results:
        clean = re.sub(r'\D', '', text)
        if 1 <= len(clean) <= 3 and conf > 0.35:
            return clean
    return None

def parse_plate_text(raw_text):
    """
    Normalize OCR output to match Indian number plate format.
    e.g., TN98AB1234, TN9846, etc.
    """
    clean = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    # Try to match standard Indian plate: 2 letters + 2 digits + 2 letters + 4 digits
    m = re.search(r'([A-Z]{2}\d{2}[A-Z]{0,2}\d{1,4})', clean)
    if m:
        return m.group(1)
    # Shorter form: 2 letters + 2-4 digits
    m = re.search(r'([A-Z]{2}\d{2,4})', clean)
    if m:
        return m.group(1)
    return clean if len(clean) >= 4 else None

def image_to_base64(image):
    """Convert OpenCV image to base64 string for storage/display."""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buffer).decode('utf-8')

# ─────────────────────────────────────────────
# BUS TRACKING LOGIC
# ─────────────────────────────────────────────
def record_bus_event(bus_id, number_plate, bus_number, snapshot_b64):
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%Y-%m-%d')

    with db_lock:
        conn = sqlite3.connect('bus_tracking.db')
        c = conn.cursor()

        c.execute('SELECT status FROM bus_status WHERE bus_id = ?', (bus_id,))
        row = c.fetchone()

        if row is None:
            # First appearance → IN campus
            c.execute('''
                INSERT INTO bus_in_campus (bus_id, number_plate, bus_number, entry_time, entry_date, image_snapshot)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (bus_id, number_plate, bus_number, time_str, date_str, snapshot_b64))
            c.execute('''
                INSERT INTO bus_status (bus_id, status, last_seen)
                VALUES (?, 'IN', ?)
            ''', (bus_id, now.isoformat()))
            event = 'IN'
        elif row[0] == 'IN':
            # Second appearance → OUT campus
            c.execute('''
                INSERT INTO bus_out_campus (bus_id, number_plate, bus_number, exit_time, exit_date, image_snapshot)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (bus_id, number_plate, bus_number, time_str, date_str, snapshot_b64))
            c.execute('UPDATE bus_status SET status=?, last_seen=? WHERE bus_id=?',
                      ('OUT', now.isoformat(), bus_id))
            event = 'OUT'
        else:
            # Subsequent appearances: cycle back IN
            c.execute('''
                INSERT INTO bus_in_campus (bus_id, number_plate, bus_number, entry_time, entry_date, image_snapshot)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (bus_id, number_plate, bus_number, time_str, date_str, snapshot_b64))
            c.execute('UPDATE bus_status SET status=?, last_seen=? WHERE bus_id=?',
                      ('IN', now.isoformat(), bus_id))
            event = 'IN'

        conn.commit()
        conn.close()
    return event

def process_image(image, mode='auto'):
    """
    Full pipeline: detect plate + bus number, record event.
    mode: 'plate' (rear), 'number' (front), 'auto' (try both)
    Returns dict with detection results.
    """
    result = {
        'success': False,
        'number_plate': None,
        'bus_number': None,
        'bus_id': None,
        'event': None,
        'message': ''
    }

    plate_text = None
    bus_number = None

    if mode in ('plate', 'auto'):
        raw, bbox = detect_number_plate(image)
        if raw:
            plate_text = parse_plate_text(raw)
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, plate_text or raw, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if mode in ('number', 'auto'):
        bus_number = detect_bus_number(image)

    if not plate_text and not bus_number:
        result['message'] = 'No bus plate or number detected. Ensure image is clear and well-lit.'
        return result, image

    result['number_plate'] = plate_text
    result['bus_number'] = bus_number

    # Build composite bus_id: e.g., TN9846-2
    if plate_text and bus_number:
        bus_id = f"{plate_text}-{bus_number}"
    elif plate_text:
        bus_id = plate_text
    else:
        bus_id = f"BUS-{bus_number}"

    result['bus_id'] = bus_id

    snapshot_b64 = image_to_base64(image)
    event = record_bus_event(bus_id, plate_text or 'N/A', bus_number or 'N/A', snapshot_b64)
    result['event'] = event
    result['success'] = True
    result['message'] = f"Bus {bus_id} recorded as {'ENTERED' if event == 'IN' else 'EXITED'} campus."

    # Draw overlay
    label = f"{bus_id} | {'IN' if event == 'IN' else 'OUT'}"
    cv2.putText(image, label, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 200, 0) if event == 'IN' else (0, 0, 255), 2)

    return result, image

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    mode = request.form.get('mode', 'auto')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    result, annotated = process_image(image, mode)

    # Encode annotated image
    annotated_b64 = image_to_base64(annotated)
    result['annotated_image'] = annotated_b64

    return jsonify(result)

@app.route('/api/bus_in')
def get_bus_in():
    conn = sqlite3.connect('bus_tracking.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT id, bus_id, number_plate, bus_number, entry_time, entry_date FROM bus_in_campus ORDER BY id DESC')
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route('/api/bus_out')
def get_bus_out():
    conn = sqlite3.connect('bus_tracking.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT id, bus_id, number_plate, bus_number, exit_time, exit_date FROM bus_out_campus ORDER BY id DESC')
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route('/api/stats')
def get_stats():
    conn = sqlite3.connect('bus_tracking.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM bus_in_campus')
    total_in = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM bus_out_campus')
    total_out = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM bus_status WHERE status='IN'")
    currently_in = c.fetchone()[0]
    conn.close()
    return jsonify({'total_in': total_in, 'total_out': total_out, 'currently_in': currently_in})

@app.route('/api/clear', methods=['POST'])
def clear_records():
    with db_lock:
        conn = sqlite3.connect('bus_tracking.db')
        c = conn.cursor()
        c.execute('DELETE FROM bus_in_campus')
        c.execute('DELETE FROM bus_out_campus')
        c.execute('DELETE FROM bus_status')
        conn.commit()
        conn.close()
    return jsonify({'message': 'All records cleared.'})

# ─────────────────────────────────────────────
# LIVE CAMERA (optional)
# ─────────────────────────────────────────────
camera_active = False
cap = None

def gen_frames():
    global cap
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while camera_active:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        # Run detection every 30 frames
        if frame_count % 30 == 0:
            process_image(frame.copy(), mode='auto')
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    if cap:
        cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/start', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    return jsonify({'status': 'Camera started'})

@app.route('/camera/stop', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({'status': 'Camera stopped'})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    init_db()
    print("🚌 Bus Tracking System starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)