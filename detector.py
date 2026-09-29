
import cv2
import time
import csv
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter, deque

try:
    from ultralytics import YOLO
except ImportError:
    print("Install first: python -m pip install ultralytics opencv-python")
    raise SystemExit(1)

# ============================================================
# AI VISION GUARDIAN
# ============================================================

MODEL_NAME = "yolo11s.pt"
CAMERA_INDEX = 0
WINDOW = "AI Vision Guardian"

MODES = {
    "FAST": {"conf": 0.45, "imgsz": 416},
    "BALANCED": {"conf": 0.60, "imgsz": 640},
    "ACCURATE": {"conf": 0.72, "imgsz": 960},
}
mode = "BALANCED"

STABLE_FRAMES = 5
MAX_MISSED = 8
SMOOTH = 0.75
VOICE_COOLDOWN = 3.0
DEMO_SECONDS = 60

BASE = Path(__file__).resolve().parent
CAPTURES = BASE / "captures"
DATA = BASE / "data"
CAPTURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
HISTORY = DATA / "detection_history.csv"

# Project-defined attention categories.
WARNING = {"knife", "scissors"}
CAUTION = {"cell phone", "laptop", "backpack", "bottle"}

# Extra components requested by the user.
# These names are supported when the loaded YOLO model contains
# matching classes.
ADDED_COMPONENTS = {
    "mouse": "mouse",
    "watch": "watch",
    "eyeglasses": "eyeglasses",
    "eye glasses": "eyeglasses",
    "pen": "pen",
    "eraser": "eraser",
}

LEVEL_COLOR = {
    "NORMAL": (0, 220, 0),
    "CAUTION": (0, 220, 255),
    "WARNING": (0, 0, 255),
}

# ------------------------------------------------------------
# History
# ------------------------------------------------------------

if not HISTORY.exists():
    with open(HISTORY, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["time", "object", "confidence", "level", "mode"]
        )

history = deque(maxlen=7)
counts = Counter()
total = 0

try:
    with open(HISTORY, newline="", encoding="utf-8") as f:
        for r in list(csv.DictReader(f))[-7:]:
            history.appendleft(r)
            counts[r["object"]] += 1
            total += 1
except Exception:
    pass

def level_for(label):
    if label in WARNING:
        return "WARNING"
    if label in CAUTION:
        return "CAUTION"
    return "NORMAL"

def save_history(label, conf, level):
    global total
    row = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "object": label,
        "confidence": f"{conf*100:.1f}",
        "level": level,
        "mode": mode,
    }
    history.appendleft(row)
    counts[label] += 1
    total += 1
    with open(HISTORY, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [row["time"], label, row["confidence"], level, mode]
        )

# ------------------------------------------------------------
# Camera
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
        time.sleep(.5)
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
    print("ERROR: Camera not available.")
    raise SystemExit(1)

print("Loading YOLO...")
model = YOLO(MODEL_NAME)
print("YOLO ready.")
print("MULTI-OBJECT MODE: ON")
print("The program will draw every object detected by the loaded model.")
print("For mouse, watch, eyeglasses, pen and eraser, load a custom YOLO")
print("model trained on those classes; yolo11s.pt cannot detect classes")
print("that are not present in its training dataset.")
print("Requested extra targets: mouse, watch, eyeglasses, pen, eraser")
print("NOTE: yolo11s.pt is normally a COCO model and does not include")
print("all of these extra classes. Use a custom YOLO model trained with")
print("these classes for actual detection of them.")

# ------------------------------------------------------------
# State
# ------------------------------------------------------------

box = None
label = None
conf = 0.0
stable = 0
missed = 0
last_voice = 0
last_spoken = ""
last_capture = "None"
fps = 0
infer_ms = 0
start_time = time.time()
demo = False
demo_start = 0

# Keep every detected object for drawing/reporting.
all_detections = []

voice_lock = threading.Lock()
voice_busy = False

def smooth_box(old, new):
    if old is None:
        return tuple(map(int, new))
    return tuple(
        int(old[i] * SMOOTH + new[i] * (1-SMOOTH))
        for i in range(4)
    )

def speak(message):
    global voice_busy
    with voice_lock:
        if voice_busy:
            return
        voice_busy = True

    def run():
        global voice_busy
        try:
            safe = message.replace("'", "''")
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate=0; $s.Volume=100; "
                f"$s.Speak('{safe}')"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8
            )
        except Exception as e:
            print("Voice error:", e)
        finally:
            voice_busy = False

    threading.Thread(target=run, daemon=True).start()

def alert(obj, confidence, level):
    global last_voice, last_spoken
    now = time.time()
    if now-last_voice < VOICE_COOLDOWN and obj == last_spoken:
        return
    last_voice = now
    last_spoken = obj

    try:
        import winsound
        if level == "WARNING":
            winsound.Beep(1500, 220)
            time.sleep(.08)
            winsound.Beep(1500, 220)
        elif level == "CAUTION":
            winsound.Beep(1100, 180)
        else:
            winsound.Beep(900, 150)
    except Exception:
        pass

    prefix = "Warning" if level == "WARNING" else (
        "Caution" if level == "CAUTION" else ""
    )
    msg = f"{prefix}. {obj} detected. Confidence {confidence*100:.0f} percent."
    print("VOICE:", msg)
    speak(msg)

def capture(frame):
    global last_capture
    if label is None or box is None:
        print("No stable object to capture.")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() else "_" for c in label)
    path = CAPTURES / f"{safe}_{conf*100:.0f}_{stamp}.jpg"
    cv2.imwrite(str(path), frame)
    last_capture = path.name
    print("Saved:", path)

def text(frame, s, xy, size=.55, color=(255,255,255), thick=1):
    cv2.putText(frame, s, xy, cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thick, cv2.LINE_AA)

def panel(frame, a, b, c, d):
    cv2.rectangle(frame, (a,b), (c,d), (18,18,18), -1)
    cv2.rectangle(frame, (a,b), (c,d), (75,75,75), 1)

# ------------------------------------------------------------
# Window
# ------------------------------------------------------------

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("""
AI VISION GUARDIAN
C = Capture
H = History
M = Change FAST/BALANCED/ACCURATE
D = Judge Demo Mode
S = Voice Test
F11 = Fullscreen
Q / ESC = Quit
""")

prev = time.time()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    frame = cv2.flip(frame, 1)

    settings = MODES[mode]
    t0 = time.perf_counter()

    results = model.predict(
        frame,
        conf=settings["conf"],
        iou=.45,
        imgsz=settings["imgsz"],
        verbose=False
    )

    infer_ms = (time.perf_counter()-t0)*1000

    candidates = []
    if results and results[0].boxes is not None:
        names = results[0].names
        for b in results[0].boxes:
            c = float(b.conf[0])
            cls = int(b.cls[0])
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            w,h = x2-x1,y2-y1
            if w >= 30 and h >= 30:
                candidates.append(
                    (c, w*h, names[cls], (x1,y1,w,h))
                )

    # MULTI-OBJECT DETECTION:
    # Keep all sufficiently confident detections instead of displaying only
    # the strongest object. The strongest object is still used for the
    # existing stable/voice/history logic.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    all_detections = candidates[:]

    if candidates:
        new_conf, _, new_label, new_box = candidates[0]
        new_label = ADDED_COMPONENTS.get(new_label, new_label)

        if label == new_label:
            box = smooth_box(box, new_box)
            conf = conf*.70 + new_conf*.30
            stable += 1
        else:
            label = new_label
            box = new_box
            conf = new_conf
            stable = 1
        missed = 0
    else:
        missed += 1
        if missed > MAX_MISSED:
            label = None
            box = None
            conf = 0
            stable = 0
            missed = 0

    is_stable = (
        label is not None and box is not None and
        stable >= STABLE_FRAMES and missed <= 2
    )

    current_level = level_for(label) if label else "NORMAL"

    if is_stable and stable == STABLE_FRAMES:
        save_history(label, conf, current_level)
        alert(label, conf, current_level)

    # Draw ALL detected objects.
    # The default COCO model can only detect its trained classes. For
    # mouse/watch/eyeglasses/pen/eraser, use a custom model containing them.
    for det_conf, _, det_label, (dx, dy, dw, dh) in all_detections:
        det_label = ADDED_COMPONENTS.get(det_label, det_label)
        det_level = level_for(det_label)
        det_color = LEVEL_COLOR[det_level]

        cv2.rectangle(frame, (dx, dy), (dx+dw, dy+dh), det_color, 2)
        det_tag = f"{det_label.upper()} {det_conf*100:.0f}%"
        (dtw, dth), _ = cv2.getTextSize(
            det_tag, cv2.FONT_HERSHEY_SIMPLEX, .55, 1
        )
        tag_y = max(dth + 6, dy)
        cv2.rectangle(
            frame, (dx, tag_y-dth-6), (dx+dtw+10, tag_y),
            det_color, -1
        )
        cv2.putText(
            frame, det_tag, (dx+5, tag_y-4),
            cv2.FONT_HERSHEY_SIMPLEX, .55, (0,0,0), 1, cv2.LINE_AA
        )

    # Draw primary stable object with the stronger visual.
    if is_stable:
        x,y,w,h = box
        color = LEVEL_COLOR[current_level]
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,4)
        tag = f"{label.upper()} {conf*100:.0f}%"
        (tw,th),_ = cv2.getTextSize(
            tag, cv2.FONT_HERSHEY_SIMPLEX, .8, 2
        )
        top = max(th+18,y)
        cv2.rectangle(frame,(x,top-th-18),(x+tw+16,top),color,-1)
        cv2.putText(frame,tag,(x+8,top-7),
                    cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2,cv2.LINE_AA)
        text(frame,f"STATUS: {current_level}",
             (x,min(frame.shape[0]-15,y+h+28)),.62,color,2)
    elif label is not None and box is not None:
        x,y,w,h = box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,200,0),2)
        text(frame,"VERIFYING...",(x,max(25,y-8)),.6,(255,200,0),2)

    now = time.time()
    fps = .9*fps + .1*(1/max(now-prev,.001))
    prev = now

    # ========================================================
    # FULL-SCREEN CAMERA UI
    # ========================================================
    # IMPORTANT: no large opaque panels. The camera stays
    # completely visible. Information is drawn on transparent
    # black strips over the image.
    # ========================================================

    H, W = frame.shape[:2]

    # Small top-left transparent status strip.
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (12, 12),
        (390, 112),
        (0, 0, 0),
        -1
    )
    frame = cv2.addWeighted(
        overlay, 0.45,
        frame, 0.55,
        0
    )

    text(
        frame,
        "AI VISION GUARDIAN",
        (25, 38),
        .68,
        (0, 255, 255),
        2
    )

    text(
        frame,
        f"OBJECT: {label.upper() if is_stable else 'SEARCHING...'}",
        (25, 64),
        .52
    )

    text(
        frame,
        f"CONF: {conf*100:.0f}%   OBJECTS: {len(all_detections)}   "
        f"FPS: {fps:.1f}",
        (25, 89),
        .46
    )

    text(
        frame,
        f"MODE: {mode}   CAMERA: ONLINE",
        (25, 108),
        .42,
        (0, 255, 0)
    )

    # Small top-right statistics strip.
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (W - 270, 12),
        (W - 12, 112),
        (0, 0, 0),
        -1
    )
    frame = cv2.addWeighted(
        overlay, 0.40,
        frame, 0.60,
        0
    )

    most = counts.most_common(1)[0][0] if counts else "None"

    text(
        frame,
        "LIVE STATS",
        (W - 250, 38),
        .60,
        (0, 255, 255),
        2
    )

    text(
        frame,
        f"Detections: {total}",
        (W - 250, 63),
        .46
    )

    text(
        frame,
        f"Most seen: {most[:18]}",
        (W - 250, 85),
        .46
    )

    text(
        frame,
        "VOICE: ON",
        (W - 250, 105),
        .42,
        (0, 255, 0)
    )

    # Demo mode.
    if demo:
        remaining=max(0,DEMO_SECONDS-(time.time()-demo_start))
        cv2.rectangle(frame,(W//2-220,15),(W//2+220,65),(80,40,180),-1)
        text(frame,f"JUDGE DEMO MODE  {remaining:.0f}s",
             (W//2-195,48),.72,(255,255,255),2)
        if remaining <= 0:
            demo=False

    # Small bottom control strip. Does not block the camera.
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, H - 42),
        (W, H),
        (0, 0, 0),
        -1
    )
    frame = cv2.addWeighted(
        overlay, 0.30,
        frame, 0.70,
        0
    )

    text(
        frame,
        "C Capture   H History   M Mode   D Demo   "
        "S Voice Test   Q Quit",
        (25, H - 15),
        .45,
        (235, 235, 235),
        1
    )

    cv2.imshow(WINDOW,frame)

    key=cv2.waitKey(1)&0xFF

    if key in (ord("q"),27):
        break

    elif key == ord("c"):
        capture(frame)

    elif key == ord("h"):
        print("\n========== DETECTION HISTORY ==========")
        for r in history:
            print(r)
        print("========================================\n")

    elif key == ord("m"):
        modes=list(MODES)
        mode=modes[(modes.index(mode)+1)%len(modes)]
        print("AI MODE:",mode)

    elif key == ord("d"):
        demo=True
        demo_start=time.time()
        print("JUDGE DEMO MODE STARTED")

    elif key == ord("s"):
        print("Voice test")
        alert("test object",.99,"NORMAL")

    elif key == 0x7A:
        # F11 toggles fullscreen if needed.
        state = cv2.getWindowProperty(
            WINDOW,
            cv2.WND_PROP_FULLSCREEN
        )
        cv2.setWindowProperty(
            WINDOW,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL
            if state == cv2.WINDOW_FULLSCREEN
            else cv2.WINDOW_FULLSCREEN
        )

cap.release()
cv2.destroyAllWindows()
print("AI Vision Guardian closed.")
print("Captures:",CAPTURES)
print("History:",HISTORY)
