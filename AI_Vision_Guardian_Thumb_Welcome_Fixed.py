import cv2
import time
import os
import csv
import subprocess
import threading
import urllib.request
import wave
import math
import struct
import re
from pathlib import Path
from datetime import datetime
from collections import Counter, deque

# ============================================================
# AI VISION GUARDIAN - SCIENCE EXPO PRO
# Object Detection + Hand Gesture Control
# Detection Sound + Evidence Capture + Live Statistics
# ============================================================

try:
    from ultralytics import YOLO
except ImportError:
    print("Install first: python -m pip install ultralytics opencv-python")
    raise SystemExit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Install first: python -m pip install mediapipe")
    raise SystemExit(1)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_NAME = "yolov8s-world.pt"
CAMERA_INDEX = 0                 # Change to 1/2/3 for external webcam
WINDOW = "AI Vision Guardian - Science Expo PRO"

MODES = {
    "FAST": {"conf": 0.22, "imgsz": 640},
    "BALANCED": {"conf": 0.15, "imgsz": 832},
    "ACCURATE": {"conf": 0.10, "imgsz": 1024},
}
mode = "BALANCED"

STABLE_FRAMES = 5
MAX_MISSED = 8
SMOOTH = 0.75
DEMO_SECONDS = 90
DETECTION_SOUND_COOLDOWN = 2.5
VOICE_COOLDOWN = 3.0
MIC_ENABLED = True
MIC_DEVICE = ""  # Leave blank to auto-select the first Windows microphone.
FFMPEG_EXE = "ffmpeg"
RECORD_FPS = 20

CAMERA_HFOV_DEG = 70.0
REFERENCE_WIDTH_M = {
    "person": 0.45,
    "laptop": 0.32,
    "computer": 0.45,
    "monitor": 0.50,
    "tablet": 0.20,
    "mobile phone": 0.075,
    "smartphone": 0.075,
    "bottle": 0.07,
    "water bottle": 0.07,
    "cup": 0.08,
    "mug": 0.10,
    "book": 0.18,
    "backpack": 0.30,
    "keyboard": 0.43,
    "mouse": 0.065,
}

LOW_LIGHT = False
HAND_GESTURE_CONTROL = True
VOICE_ENABLED = True
SHOW_LANDMARKS = True
MONITORING = True

BASE = Path(__file__).resolve().parent
CAPTURES = BASE / "captures"
DATA = BASE / "data"
CAPTURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
HISTORY = DATA / "detection_history.csv"
EVENT_LOG = DATA / "event_log.csv"

WARNING = set()
CAUTION = {"cell phone", "mobile phone", "smartphone", "laptop", "backpack", "bottle"}

ADDED_COMPONENTS = {
    "mouse": "mouse",
    "computer mouse": "mouse",
    "watch": "watch",
    "wrist watch": "watch",
    "eyeglasses": "eyeglasses",
    "glasses": "eyeglasses",
    "sunglasses": "sunglasses",
    "pen": "pen",
    "ballpoint pen": "pen",
    "eraser": "eraser",
}

TARGET_CLASSES = [
    "computer mouse", "wrist watch", "watch",
    "eyeglasses", "glasses", "sunglasses",
    "pen", "ballpoint pen", "eraser", "pencil", "pencil sharpener",
    "book", "notebook", "paper", "document",
    "mobile phone", "smartphone", "tablet", "keyboard", "laptop",
    "computer", "monitor", "charger", "USB cable", "headphones",
    "bottle", "cup", "mug", "water bottle", "backpack", "wallet",
    "keys", "remote control", "calculator", "ruler", "stapler",
    "fork", "spoon",
    "person", "chair", "table", "clock",
]

LEVEL_COLOR = {
    "NORMAL": (0, 220, 0),
    "CAUTION": (0, 220, 255),
    "WARNING": (0, 0, 255),
}

if not HISTORY.exists():
    with open(HISTORY, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["time", "object", "confidence", "level", "mode"])

if not EVENT_LOG.exists():
    with open(EVENT_LOG, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "time", "event", "object", "confidence", "distance_m",
            "position", "mode"
        ])

history = deque(maxlen=10)
counts = Counter()
total = 0
warnings = 0

try:
    with open(HISTORY, newline="", encoding="utf-8") as f:
        for r in list(csv.DictReader(f))[-10:]:
            history.appendleft(r)
            counts[r["object"]] += 1
            total += 1
except Exception:
    pass


def text(frame, message, org, scale=0.5, color=(235, 235, 235), thickness=1):
    cv2.putText(
        frame, str(message), org, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


def level_for(label):
    if label in WARNING:
        return "WARNING"
    if label in CAUTION:
        return "CAUTION"
    return "NORMAL"


def log_event(event, obj="", confidence=0.0, distance=None, position=""):
    with open(EVENT_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event, obj, f"{confidence * 100:.1f}",
            "" if distance is None else f"{distance:.2f}",
            position, mode
        ])


def save_history(label, conf, level):
    global total
    row = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "object": label,
        "confidence": f"{conf * 100:.1f}",
        "level": level,
        "mode": mode,
    }
    history.appendleft(row)
    counts[label] += 1
    total += 1
    with open(HISTORY, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row["time"], label, row["confidence"], level, mode])

# ------------------------------------------------------------
# MediaPipe hand gesture model setup
# ------------------------------------------------------------
HAND_MODEL = BASE / "gesture_recognizer.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

def ensure_model(path, url, name):
    if path.exists() and path.stat().st_size > 10000:
        return True
    print(f"{name} not found. Attempting official model download...")
    try:
        urllib.request.urlretrieve(url, str(path))
        print(f"Downloaded: {path}")
        return path.exists() and path.stat().st_size > 10000
    except Exception as e:
        print(f"Could not download {name}: {e}")
        return False

if not ensure_model(HAND_MODEL, HAND_MODEL_URL, "gesture_recognizer.task"):
    raise SystemExit(4)

hand_options = vision.GestureRecognizerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.45,
    min_hand_presence_confidence=0.45,
    min_tracking_confidence=0.45,
)
hand_recognizer = vision.GestureRecognizer.create_from_options(hand_options)

hand_timestamp_ms = 0


def process_hands(frame):
    global hand_timestamp_ms
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hand_timestamp_ms += 33
    result = hand_recognizer.recognize_for_video(image, hand_timestamp_ms)
    hands = []
    h, w = frame.shape[:2]
    if result.gestures and result.hand_landmarks:
        for i, landmarks in enumerate(result.hand_landmarks[:2]):
            gesture = "None"
            score = 0.0
            handed = "Hand"
            if i < len(result.gestures) and result.gestures[i]:
                cat = result.gestures[i][0]
                gesture = cat.category_name or cat.display_name or "None"
                score = float(cat.score)
            if i < len(result.handedness) and result.handedness[i]:
                handed = result.handedness[i][0].display_name or "Hand"
            pts = [(p.x * w, p.y * h) for p in landmarks]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            box = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            center = (sum(xs) / len(xs), sum(ys) / len(ys))
            hands.append({
                "gesture": gesture,
                "score": score,
                "handed": handed,
                "box": box,
                "center": center,
                "landmarks": pts,
            })
    return hands

# ------------------------------------------------------------
# Camera / YOLO
# ------------------------------------------------------------

def open_camera():
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        time.sleep(0.4)
        good = 0
        for _ in range(4):
            ok, f = cap.read()
            good += int(ok and f is not None and f.size > 0)
        if good >= 2:
            return cap
        cap.release()
    return None


cap = open_camera()
if cap is None:
    print(f"ERROR: Camera index {CAMERA_INDEX} is not available.")
    raise SystemExit(1)

LOCAL_MODEL = BASE / MODEL_NAME
if not LOCAL_MODEL.exists():
    print("ERROR: Missing local model:", LOCAL_MODEL)
    cap.release()
    raise SystemExit(2)

print("Loading YOLO-World...")
model = YOLO(str(LOCAL_MODEL))
try:
    model.set_classes(TARGET_CLASSES)
except Exception as e:
    print("WARNING: YOLO-World text classes failed:", e)
    cap.release()
    raise
print("YOLO-World ready.")

# ------------------------------------------------------------
# State Variables
# ------------------------------------------------------------
box = None
label = None
conf = 0.0
stable = 0
missed = 0
last_capture = "None"
fps = 0.0
infer_ms = 0.0
start_time = time.time()
demo = False
demo_start = 0.0
all_detections = []
tracked_objects = {}
next_object_id = 1
hands = []
hand_count = 0
recording = False
video_writer = None
record_process = None
record_pipe = None
recording_audio_device = None
last_announced_objects = {}
last_sound_times = {}
voice_seen_objects = set()
active_speech_process = None
active_welcome_process = None
voice_missing_frames = {}
VOICE_REARM_FRAMES = 12
WELCOME_COOLDOWN = 5.0
last_welcome_time = 0.0
last_gesture_action = 0.0
last_gesture_name = "NONE"

GESTURE_STABLE_FRAMES = 3
gesture_candidate = "NONE"
gesture_candidate_frames = 0


def smooth_box(old, new):
    if old is None:
        return tuple(map(int, new))
    return tuple(int(old[i] * SMOOTH + new[i] * (1 - SMOOTH)) for i in range(4))


DETECTION_WAV = DATA / "object_detected.wav"
ACTION_WAV = DATA / "gesture_action.wav"

def make_tone_wav(path, frequencies, duration_ms=180, volume=0.28):
    try:
        sample_rate = 44100
        total = int(sample_rate * duration_ms / 1000)
        part_len = max(1, total // len(frequencies))
        frames = []
        for i in range(total):
            freq = frequencies[min(i // part_len, len(frequencies) - 1)]
            fade = min(1.0, i / max(1, sample_rate * 0.008),
                       (total - i) / max(1, sample_rate * 0.012))
            value = int(32767 * volume * fade *
                        math.sin(2 * math.pi * freq * i / sample_rate))
            frames.append(struct.pack("<h", value))
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        return True
    except Exception as e:
        print("Could not create sound:", e)
        return False

if not DETECTION_WAV.exists():
    make_tone_wav(DETECTION_WAV, (880, 1320), 180, 0.30)
if not ACTION_WAV.exists():
    make_tone_wav(ACTION_WAV, (660,), 110, 0.25)

def play_wav(path):
    try:
        import winsound
        winsound.PlaySound(
            str(path),
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
        )
        return True
    except Exception as e:
        print("Sound playback error:", e)
        return False

def detection_sound(label=None):
    global active_speech_process
    if not VOICE_ENABLED:
        return

    play_wav(DETECTION_WAV)
    if not label:
        return

    safe_label = str(label).replace('"', "'")

    def _speak():
        global active_speech_process
        try:
            if active_speech_process is not None and active_speech_process.poll() is None:
                active_speech_process.terminate()

            ps = (
                'Add-Type -AssemblyName System.Speech; '
                '$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                '$s.Rate=0; '
                '$s.Volume=90; '
                f'$s.Speak("Object detected: {safe_label}")'
            )
            active_speech_process = subprocess.Popen(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        except Exception as e:
            print("Voice playback error:", e)

    threading.Thread(target=_speak, daemon=True).start()

def welcome_voice():
    global active_welcome_process
    if not VOICE_ENABLED:
        return

    # OPTION 3 WELCOME MESSAGE
    message = (
        "Welcome to the Computer Vision and AI Exhibition. "
        "Topic: Real-Time Object Detection and Gesture Recognition. "
        "AI Guardian System is now active and ready."
    )

    def _speak_welcome():
        global active_welcome_process
        try:
            if active_welcome_process is not None and active_welcome_process.poll() is None:
                active_welcome_process.terminate()

            safe_message = message.replace('"', "'")
            ps = (
                'Add-Type -AssemblyName System.Speech; '
                '$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                '$s.Rate=0; '
                '$s.Volume=100; '
                f'$s.Speak("{safe_message}"); '
                '$s.Dispose()'
            )

            active_welcome_process = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )

            print("EXPO WELCOME VOICE STARTED")
            _, err = active_welcome_process.communicate(timeout=20)
            if active_welcome_process.returncode != 0 and err:
                print("Welcome voice error:", err.strip())

        except Exception as e:
            print("Welcome voice error:", e)

    threading.Thread(target=_speak_welcome, daemon=True).start()


def action_sound():
    if VOICE_ENABLED:
        play_wav(ACTION_WAV)


def capture(frame, reason="manual"):
    global last_capture
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe = "".join(c if c.isalnum() else "_" for c in str(reason)) or "event"
    path = CAPTURES / f"{safe}_{stamp}.jpg"
    if cv2.imwrite(str(path), frame):
        last_capture = path.name
        print("Evidence saved:", path)
        return path
    return None


def find_microphone_device():
    if MIC_DEVICE:
        return MIC_DEVICE
    try:
        result = subprocess.run(
            [FFMPEG_EXE, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        combined = result.stderr + "\n" + result.stdout
        names = re.findall(r'"([^"]+)"\s*\(audio\)', combined, flags=re.I)
        return names[0] if names else None
    except Exception as e:
        print("Microphone detection error:", e)
        return None


def start_audio_video_recording(frame):
    global recording, record_process, record_pipe, recording_audio_device

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CAPTURES / f"session_{stamp}.mp4"
    h, w = frame.shape[:2]

    mic = find_microphone_device()
    if not mic:
        print("ERROR: No Windows microphone found.")
        return False

    cmd = [
        FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", f"{w}x{h}", "-framerate", str(RECORD_FPS),
        "-i", "-",
        "-thread_queue_size", "512",
        "-f", "dshow", "-i", f"audio={mic}",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-r", str(RECORD_FPS),
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(path)
    ]

    try:
        record_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        record_pipe = record_process.stdin
        recording_audio_device = mic
        recording = True
        print("Recording with microphone:", mic)
        return True
    except Exception as e:
        print("ERROR: Could not start audio recording:", e)
        return False


def stop_audio_video_recording():
    global recording, record_process, record_pipe, recording_audio_device
    recording = False

    if record_pipe is not None:
        try:
            record_pipe.close()
        except Exception:
            pass
        record_pipe = None

    if record_process is not None:
        try:
            record_process.wait(timeout=8)
        except Exception:
            try:
                record_process.kill()
            except Exception:
                pass
        record_process = None

    print("Recording stopped.")
    recording_audio_device = None


def toggle_recording(frame):
    if not recording:
        start_audio_video_recording(frame)
    else:
        stop_audio_video_recording()


def low_light_enhance(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def object_position(x, y, w, h, frame_width):
    cx = x + w / 2
    if cx < frame_width * 0.33:
        return "LEFT"
    elif cx > frame_width * 0.67:
        return "RIGHT"
    return "CENTER"


def estimate_distance(label_, pixel_width, frame_width):
    ref = REFERENCE_WIDTH_M.get(label_)
    if not ref or pixel_width <= 3:
        return None
    focal_px = (frame_width / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG / 2.0))
    d = (ref * focal_px) / pixel_width
    return max(0.15, min(d, 30.0))


def distance_text(d):
    if d is None:
        return "DIST: --"
    return f"DIST: {d:.2f}m" if d < 1 else f"DIST: {d:.1f}m"


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def update_tracking(detections):
    global tracked_objects, next_object_id
    updated = {}
    used = set()
    for conf_, _, label_, box_ in detections:
        best_id, best_iou = None, 0.0
        for oid, obj in tracked_objects.items():
            if oid in used or obj["label"] != label_:
                continue
            score = bbox_iou(obj["box"], box_)
            if score > best_iou:
                best_iou, best_id = score, oid
        if best_id is None or best_iou < 0.20:
            best_id = next_object_id
            next_object_id += 1
        updated[best_id] = {
            "label": label_, "conf": conf_, "box": box_, "seen": time.time()
        }
        used.add(best_id)
    tracked_objects = updated


def draw_hand_landmarks(frame, hand):
    if not SHOW_LANDMARKS:
        return
    pts = hand["landmarks"]
    for p in pts:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (255, 170, 0), -1)
    for a, b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                 (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
        p1, p2 = pts[a], pts[b]
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 170, 0), 1)


def draw_hand_panel(frame, hand_list):
    H, W = frame.shape[:2]
    x0 = max(10, W - 365)
    y0 = min(H - 170, 300)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (W - 12, y0 + 125), (5, 5, 5), -1)
    frame[:] = cv2.addWeighted(overlay, .50, frame, .50, 0)
    text(frame, "HANDS-FREE CONTROL", (x0 + 18, y0 + 28), .55, (255, 170, 0), 2)
    if not hand_list:
        text(frame, "No hand detected", (x0 + 18, y0 + 55), .42)
    for i, hnd in enumerate(hand_list[:2]):
        text(frame, f"{hnd['handed']}: {hnd['gesture']} {hnd['score']*100:.0f}%", (x0 + 18, y0 + 55 + i * 24), .42)
    text(frame, "Thumb Up=welcome | Palm=pause | Victory=capture | Point=mode", (x0 + 18, y0 + 103), .34, (220,220,220), 1)


def draw_dashboard(frame):
    H, W = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (445, 145), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, .48, frame, .52, 0)
    primary = label.upper() if label and stable >= STABLE_FRAMES else "SEARCHING"
    text(frame, "AI VISION GUARDIAN", (25, 38), .68, (0, 255, 255), 2)
    text(frame, f"OBJECT: {primary}   CONF: {conf*100:.0f}%", (25, 64), .48)
    text(frame, f"OBJECTS: {len(all_detections)}   HANDS: {hand_count}", (25, 88), .45)
    text(frame, f"FPS: {fps:.1f}   INFER: {infer_ms:.0f}ms   MODE: {mode}", (25, 112), .43)
    status = "PAUSED" if not MONITORING else "ACTIVE"
    status_color = (0, 220, 255) if not MONITORING else (0, 255, 0)
    x0 = W - 285
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 12), (W - 12, 145), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, .48, frame, .52, 0)
    most = counts.most_common(1)[0][0] if counts else "None"
    text(frame, "EXPO LIVE DASHBOARD", (x0 + 15, 38), .55, (0,255,255), 2)
    text(frame, f"SYSTEM: {status}", (x0 + 15, 64), .44, status_color, 2)
    text(frame, f"Total detections: {total}", (x0 + 15, 87), .40)
    text(frame, f"Most seen: {most[:19]}", (x0 + 15, 108), .40)
    text(frame, "OBJECT SOUND + VOICE: ON", (x0 + 15, 130), .36, (0,255,0), 2)


def normalize_gesture_name(gesture):
    g = str(gesture or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "thumb_up": "Thumb_Up",
        "thumbup": "Thumb_Up",
        "thumbs_up": "Thumb_Up",
        "thumbs_up_gesture": "Thumb_Up",
        "thumb_down": "Thumb_Down",
        "thumbdown": "Thumb_Down",
        "thumbs_down": "Thumb_Down",
        "open_palm": "Open_Palm",
        "openhand": "Open_Palm",
        "open_hand": "Open_Palm",
        "victory": "Victory",
        "peace": "Victory",
        "pointing_up": "Pointing_Up",
        "point_up": "Pointing_Up",
        "i_love_you": "ILoveYou",
        "iloveyou": "ILoveYou",
    }
    return aliases.get(g, gesture)


def perform_hand_action(gesture, frame):
    global MONITORING, mode, LOW_LIGHT
    global last_gesture_action, last_gesture_name, last_welcome_time

    now = time.time()
    gesture = normalize_gesture_name(gesture)

    if not HAND_GESTURE_CONTROL or gesture in ("None", ""):
        return

    last_gesture_name = gesture

    if gesture == "Thumb_Up":
        if now - last_welcome_time >= WELCOME_COOLDOWN:
            print("THUMB UP CONFIRMED -> PLAYING EXPO WELCOME")
            welcome_voice()
            last_welcome_time = now
            action_sound()
        MONITORING = True
        last_gesture_action = now
        return

    if now - last_gesture_action < 1.8:
        return

    if gesture == "Open_Palm":
        MONITORING = not MONITORING
    elif gesture == "Victory":
        capture(frame, "hand_capture")
    elif gesture == "Pointing_Up":
        modes = list(MODES)
        mode = modes[(modes.index(mode) + 1) % len(modes)]
    elif gesture in ("Thumb_Down", "ILoveYou"):
        LOW_LIGHT = not LOW_LIGHT
    else:
        return

    action_sound()
    print("HAND ACTION:", gesture)
    last_gesture_action = now

# ------------------------------------------------------------
# Window Setup
# ------------------------------------------------------------
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev = time.time()

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
try:
    while True:
        ok, raw = cap.read()
        if not ok or raw is None:
            continue

        raw = cv2.flip(raw, 1)
        inference_frame = low_light_enhance(raw) if LOW_LIGHT else raw.copy()
        H, W = inference_frame.shape[:2]

        try:
            hands = process_hands(inference_frame) if HAND_GESTURE_CONTROL else []
        except Exception as e:
            hands = []
            print("Hand processing error:", e)
        hand_count = len(hands)

        if hands:
            best_hand = max(hands, key=lambda x: x["score"])
            raw_gesture = normalize_gesture_name(best_hand["gesture"])
            score = best_hand["score"]

            min_score = 0.45 if raw_gesture == "Thumb_Up" else 0.55

            if score >= min_score and raw_gesture not in ("None", ""):
                if raw_gesture == gesture_candidate:
                    gesture_candidate_frames += 1
                else:
                    gesture_candidate = raw_gesture
                    gesture_candidate_frames = 1

                required_frames = 1 if raw_gesture == "Thumb_Up" else GESTURE_STABLE_FRAMES

                if gesture_candidate_frames >= required_frames:
                    perform_hand_action(raw_gesture, raw)
                    gesture_candidate_frames = 0
            else:
                gesture_candidate = "NONE"
                gesture_candidate_frames = 0
        else:
            gesture_candidate = "NONE"
            gesture_candidate_frames = 0

        candidates = []
        if MONITORING:
            settings = MODES[mode]
            t0 = time.perf_counter()
            try:
                results = model.predict(
                    inference_frame,
                    conf=settings["conf"],
                    iou=0.45,
                    imgsz=settings["imgsz"],
                    verbose=False,
                )
            except Exception as e:
                print("YOLO error:", e)
                results = []
            infer_ms = (time.perf_counter() - t0) * 1000

            if results and results[0].boxes is not None:
                names = results[0].names
                for b in results[0].boxes:
                    c = float(b.conf[0])
                    cls = int(b.cls[0])
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    bw, bh = x2 - x1, y2 - y1
                    if bw >= 12 and bh >= 12:
                        candidates.append((c, bw * bh, names[cls], (x1, y1, bw, bh)))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        all_detections = candidates[:20]
        update_tracking(all_detections)

        if candidates and MONITORING:
            new_conf, _, new_label, new_box = candidates[0]
            new_label = ADDED_COMPONENTS.get(new_label, new_label)
            if label == new_label:
                box = smooth_box(box, new_box)
                conf = conf * 0.70 + new_conf * 0.30
                stable += 1
            else:
                label, box, conf, stable = new_label, new_box, new_conf, 1
            missed = 0
        else:
            missed += 1
            if missed > MAX_MISSED:
                label, box, conf, stable = None, None, 0.0, 0
                missed = 0

        is_stable = bool(label and box and stable >= STABLE_FRAMES and missed <= 2 and MONITORING)
        current_level = level_for(label) if label else "NORMAL"

        if is_stable and stable == STABLE_FRAMES:
            save_history(label, conf, current_level)

        visible_voice_objects = set()
        for det_conf, _, det_label, _ in candidates[:20]:
            det_label = ADDED_COMPONENTS.get(det_label, det_label)
            if det_conf >= settings["conf"]:
                visible_voice_objects.add(det_label)

        for old_label in list(voice_seen_objects):
            if old_label not in visible_voice_objects:
                voice_missing_frames[old_label] = voice_missing_frames.get(old_label, 0) + 1
                if voice_missing_frames[old_label] >= VOICE_REARM_FRAMES:
                    voice_seen_objects.discard(old_label)
                    voice_missing_frames.pop(old_label, None)

        for det_conf, _, det_label, _ in candidates[:20]:
            det_label = ADDED_COMPONENTS.get(det_label, det_label)
            if det_conf >= settings["conf"] and det_label not in voice_seen_objects:
                detection_sound(det_label)
                voice_seen_objects.add(det_label)
                voice_missing_frames.pop(det_label, None)

        display = inference_frame.copy()
        for oid, obj in tracked_objects.items():
            dl = ADDED_COMPONENTS.get(obj["label"], obj["label"])
            dx, dy, dw, dh = obj["box"]
            level = level_for(dl)
            color = LEVEL_COLOR[level]
            d = estimate_distance(dl, dw, W)
            cv2.rectangle(display, (dx, dy), (dx + dw, dy + dh), color, 2)
            tag = f"#{oid} {dl.upper()} {obj['conf']*100:.0f}%"
            cv2.rectangle(display, (dx, max(0, dy - 26)), (min(W - 1, dx + 235), dy), color, -1)
            text(display, tag, (dx + 6, max(18, dy - 7)), .45, (0, 0, 0), 2)
            detail = f"{object_position(dx,dy,dw,dh,W)} | {distance_text(d)}"
            text(display, detail, (dx, min(H - 8, dy + dh + 18)), .40, color, 1)

        if is_stable:
            x, y, bw, bh = box
            color = LEVEL_COLOR[current_level]
            cv2.rectangle(display, (x, y), (x + bw, y + bh), color, 4)
            text(display, f"STATUS: {current_level}", (x, min(H - 15, y + bh + 42)), .55, color, 2)

        draw_hand_panel(display, hands)
        for hnd in hands:
            draw_hand_landmarks(display, hnd)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - prev, 0.001))
        prev = now
        draw_dashboard(display)

        if demo:
            remaining = max(0, DEMO_SECONDS - (time.time() - demo_start))
            cv2.rectangle(display, (W // 2 - 245, 12), (W // 2 + 245, 65), (80, 40, 180), -1)
            text(display, f"JUDGE DEMO MODE  {remaining:.0f}s", (W // 2 - 220, 48), .68, (255,255,255), 2)
            if remaining <= 0:
                demo = False

        if recording:
            cv2.circle(display, (W - 30, H - 65), 8, (0, 0, 255), -1)
            text(display, "REC", (W - 70, H - 58), .45, (0,0,255), 2)

        overlay = display.copy()
        cv2.rectangle(overlay, (0, H - 43), (W, H), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, .36, display, .64, 0)
        text(display,
             "C Capture  R Record  M Mode  D Demo  L LowLight  G Gestures  F Landmarks  Q Quit",
             (18, H - 15), .39, (235,235,235), 1)

        if recording and record_pipe is not None:
            try:
                record_pipe.write(display.tobytes())
            except (BrokenPipeError, OSError):
                stop_audio_video_recording()

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            capture(raw, "manual")
        elif key == ord("r"):
            toggle_recording(display)
        elif key == ord("m"):
            modes = list(MODES)
            mode = modes[(modes.index(mode) + 1) % len(modes)]
        elif key == ord("w"):
            welcome_voice()
        elif key == ord("d"):
            demo = True
            demo_start = time.time()
        elif key == ord("l"):
            LOW_LIGHT = not LOW_LIGHT
        elif key == ord("g"):
            HAND_GESTURE_CONTROL = not HAND_GESTURE_CONTROL
        elif key == ord("f"):
            SHOW_LANDMARKS = not SHOW_LANDMARKS
        elif key == 0x7A:
            state = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL if state == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
            )
finally:
    cap.release()
    if recording:
        stop_audio_video_recording()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    try:
        hand_recognizer.close()
    except Exception:
        pass