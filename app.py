from flask import Flask, render_template, request, jsonify
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
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

reader = easyocr.Reader(['en'], verbose=False)
db_lock = threading.Lock()

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('bus_tracking.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bus_in_campus (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bus_id TEXT NOT NULL,
        number_plate TEXT NOT NULL,
        bus_number TEXT NOT NULL,
        entry_time TEXT NOT NULL,
        entry_date TEXT NOT NULL,
        front_snapshot TEXT,
        rear_snapshot TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS bus_out_campus (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bus_id TEXT NOT NULL,
        number_plate TEXT NOT NULL,
        bus_number TEXT NOT NULL,
        exit_time TEXT NOT NULL,
        exit_date TEXT NOT NULL,
        front_snapshot TEXT,
        rear_snapshot TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS bus_status (
        bus_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        last_seen TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────
def preprocess_for_ocr(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = max(1, 100 // max(h, 1))
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def image_to_base64(image):
    _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return base64.b64encode(buf).decode('utf-8')

# ─────────────────────────────────────────────
# DETECTION: NUMBER PLATE (REAR)
# ─────────────────────────────────────────────
def detect_number_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 20))

    best_text, bbox = "", None

    if len(plates) > 0:
        plates = sorted(plates, key=lambda p: p[2]*p[3], reverse=True)
        x, y, w, h = plates[0]
        pad = 10
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(image.shape[1], x+w+pad), min(image.shape[0], y+h+pad)
        roi = image[y1:y2, x1:x2]
        processed = preprocess_for_ocr(roi)
        results = reader.readtext(processed, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        for (_, text, conf) in results:
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(clean) >= 4:
                best_text = clean
                bbox = (x1, y1, x2, y2)
                break

    # Fallback: scan image halves
    if not best_text:
        h_img = image.shape[0]
        for region in [image[:h_img//2, :], image[h_img//2:, :]]:
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

# ─────────────────────────────────────────────
# DETECTION: BUS ROUTE NUMBER (FRONT)
# ─────────────────────────────────────────────
def detect_bus_number(image):
    h, w = image.shape[:2]
    roi = image[:int(h*0.45), w//5: 4*w//5]
    processed = preprocess_for_ocr(roi)
    results = reader.readtext(processed, detail=1, allowlist='0123456789')
    for (_, text, conf) in results:
        clean = re.sub(r'\D', '', text)
        if 1 <= len(clean) <= 3 and conf > 0.4:
            return clean
    # Full image fallback
    results = reader.readtext(image, detail=1, allowlist='0123456789')
    for (_, text, conf) in results:
        clean = re.sub(r'\D', '', text)
        if 1 <= len(clean) <= 3 and conf > 0.35:
            return clean
    return None

def parse_plate_text(raw):
    clean = re.sub(r'[^A-Z0-9]', '', raw.upper())
    m = re.search(r'([A-Z]{2}\d{2}[A-Z]{0,2}\d{1,4})', clean)
    if m: return m.group(1)
    m = re.search(r'([A-Z]{2}\d{2,4})', clean)
    if m: return m.group(1)
    return clean if len(clean) >= 4 else None

# ─────────────────────────────────────────────
# RECORD EVENT IN DB (scan count logic)
# ─────────────────────────────────────────────
def record_bus_event(bus_id, number_plate, bus_number, front_b64, rear_b64):
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%Y-%m-%d')

    with db_lock:
        conn = sqlite3.connect('bus_tracking.db')
        c = conn.cursor()
        c.execute('SELECT status FROM bus_status WHERE bus_id = ?', (bus_id,))
        row = c.fetchone()

        if row is None:
            # 1st scan → IN
            c.execute('''INSERT INTO bus_in_campus
                (bus_id, number_plate, bus_number, entry_time, entry_date, front_snapshot, rear_snapshot)
                VALUES (?,?,?,?,?,?,?)''',
                (bus_id, number_plate, bus_number, time_str, date_str, front_b64, rear_b64))
            c.execute("INSERT INTO bus_status (bus_id, status, last_seen) VALUES (?, 'IN', ?)",
                      (bus_id, now.isoformat()))
            event = 'IN'

        elif row[0] == 'IN':
            # 2nd scan → OUT
            c.execute('''INSERT INTO bus_out_campus
                (bus_id, number_plate, bus_number, exit_time, exit_date, front_snapshot, rear_snapshot)
                VALUES (?,?,?,?,?,?,?)''',
                (bus_id, number_plate, bus_number, time_str, date_str, front_b64, rear_b64))
            c.execute("UPDATE bus_status SET status='OUT', last_seen=? WHERE bus_id=?",
                      (now.isoformat(), bus_id))
            event = 'OUT'

        else:
            # Cycle back → IN
            c.execute('''INSERT INTO bus_in_campus
                (bus_id, number_plate, bus_number, entry_time, entry_date, front_snapshot, rear_snapshot)
                VALUES (?,?,?,?,?,?,?)''',
                (bus_id, number_plate, bus_number, time_str, date_str, front_b64, rear_b64))
            c.execute("UPDATE bus_status SET status='IN', last_seen=? WHERE bus_id=?",
                      (now.isoformat(), bus_id))
            event = 'IN'

        conn.commit()
        conn.close()
    return event

# ─────────────────────────────────────────────
# DUAL-IMAGE PROCESSING PIPELINE
# ─────────────────────────────────────────────
def process_dual_images(front_image, rear_image):
    result = {
        'success': False,
        'number_plate': None,
        'bus_number': None,
        'bus_id': None,
        'event': None,
        'message': '',
        'front_annotated': None,
        'rear_annotated': None,
    }
    warnings = []

    # ── REAR → detect number plate ──
    plate_text = None
    if rear_image is not None:
        raw, bbox = detect_number_plate(rear_image)
        if raw:
            plate_text = parse_plate_text(raw)
            # Draw bounding box
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(rear_image, (x1, y1), (x2, y2), (0, 255, 80), 2)
                cv2.putText(rear_image, plate_text or raw,
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
            cv2.putText(rear_image, f"PLATE: {plate_text or raw}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 80), 2)
        else:
            warnings.append('Plate not detected in rear photo.')

    # ── FRONT → detect route number ──
    bus_number = None
    if front_image is not None:
        bus_number = detect_bus_number(front_image)
        if bus_number:
            cv2.putText(front_image, f"ROUTE: {bus_number}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)
        else:
            warnings.append('Route number not detected in front photo.')

    # ── Need at least one detection ──
    if not plate_text and not bus_number:
        result['message'] = 'Detection failed on both photos. Ensure images are clear and well-lit.'
        result['front_annotated'] = image_to_base64(front_image) if front_image is not None else None
        result['rear_annotated']  = image_to_base64(rear_image)  if rear_image  is not None else None
        return result

    result['number_plate'] = plate_text
    result['bus_number']   = bus_number

    # ── Build Bus ID: TN9846-2 ──
    if plate_text and bus_number:
        bus_id = f"{plate_text}-{bus_number}"
    elif plate_text:
        bus_id = plate_text
    else:
        bus_id = f"BUS-{bus_number}"

    result['bus_id'] = bus_id

    # ── Save to DB ──
    front_b64 = image_to_base64(front_image) if front_image is not None else None
    rear_b64  = image_to_base64(rear_image)  if rear_image  is not None else None

    event = record_bus_event(
        bus_id,
        plate_text  or 'N/A',
        bus_number  or 'N/A',
        front_b64,
        rear_b64
    )

    result['event']   = event
    result['success'] = True
    status_word       = 'ENTERED' if event == 'IN' else 'EXITED'
    warn_str          = (' | ' + ' | '.join(warnings)) if warnings else ''
    result['message'] = f"Bus {bus_id} {status_word} campus.{warn_str}"

    # ── Stamp event label on images ──
    color = (0, 220, 100) if event == 'IN' else (0, 60, 255)
    stamp = f"{bus_id}  {'>>> IN CAMPUS' if event == 'IN' else '<<< OUT CAMPUS'}"
    for img in [front_image, rear_image]:
        if img is not None:
            cv2.putText(img, stamp, (10, img.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    result['front_annotated'] = image_to_base64(front_image) if front_image is not None else None
    result['rear_annotated']  = image_to_base64(rear_image)  if rear_image  is not None else None
    return result

# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    front_image = None
    rear_image  = None

    if 'front' in request.files and request.files['front'].filename != '':
        fb = np.frombuffer(request.files['front'].read(), np.uint8)
        front_image = cv2.imdecode(fb, cv2.IMREAD_COLOR)

    if 'rear' in request.files and request.files['rear'].filename != '':
        rb = np.frombuffer(request.files['rear'].read(), np.uint8)
        rear_image = cv2.imdecode(rb, cv2.IMREAD_COLOR)

    if front_image is None and rear_image is None:
        return jsonify({'error': 'Upload at least one photo (front or rear).'}), 400

    result = process_dual_images(front_image, rear_image)
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

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    init_db()
    print("🚌 Bus Tracking System → http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)