"""
YOLO + tracker: per-person speed estimation with a latched "fast" flag.

- Every person gets a persistent track ID from the tracker.
- Speed is measured on the person's foot point over a short time window.
- YELLOW = not (yet) above the threshold.
- GREEN  = was above the threshold at some point, and stays green for the rest of the video.

Usage:  from nbutils.tracker import track_speed
        track_speed(videos[0], "out/speed_0.mp4", threshold=2.0)

Weights are read from models/ (git-ignored); ultralytics downloads them
automatically on first use if they are missing.
"""
import math
from collections import defaultdict, deque
from pathlib import Path
from statistics import median

import cv2
from ultralytics import YOLO

GREEN = (0, 255, 0)    # BGR
YELLOW = (0, 255, 255)


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def measure_speed(hist, fps, unit, person_height_m):
    """Speed over `hist`, a deque of (frame_idx, x, y, box_height) samples."""
    f0, x0, y0, _ = hist[0]
    f1, x1, y1, _ = hist[-1]
    dt = (f1 - f0) / fps                                   # seconds, from frame numbers
    if dt <= 0:
        return None
    speed = math.hypot(x1 - x0, y1 - y0) / dt              # pixels / second
    if unit == "m":                                        # scale by body height -> approx. m/s
        speed = speed / median(s[3] for s in hist) * person_height_m
    return speed


def draw_label(frame, x, y, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, 0.5, 1)
    top = max(y - th - base - 4, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 6, top + th + base + 4), color, -1)
    cv2.putText(frame, text, (x + 3, top + th + 2), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def track_speed(
    path,
    out_path="speed.mp4",
    threshold=4.0,            # speed above which a person turns green (in `unit`)
    unit="m",                 # "m" = approx. metres/second, "px" = pixels/second
    person_height_m=1.7,      # only used when unit="m"
    window_s=0.5,             # speed is measured over this many seconds
    confirm_frames=3,         # consecutive frames above threshold before latching green
    max_gap=15,               # frames; a track missing longer than this loses its history
    model_name="models/yolov8m.pt",
    tracker="tracker.yaml",
    conf=0.15, iou=0.8, imgsz=960,
):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    window = max(2, round(fps * window_s))                 # window length in frames
    unit_label = "m/s" if unit == "m" else "px/s"

    Path(model_name).parent.mkdir(parents=True, exist_ok=True)   # so ultralytics can fetch it there
    model = YOLO(model_name)                               # new instance = fresh tracker state
    device = get_device()
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    history = defaultdict(lambda: deque(maxlen=window + 1))  # id -> (frame, x, foot_y, box_h)
    above = defaultdict(int)                                 # id -> consecutive frames above threshold
    fast_ids = set()                                         # latched green

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            r = model.track(
                frame, persist=True, tracker=tracker,
                classes=[0], conf=conf, iou=iou, imgsz=imgsz,
                device=device, verbose=False,
            )[0]

            n_now = 0
            if r.boxes.id is not None:
                ids = r.boxes.id.int().cpu().tolist()
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                n_now = len(ids)

                for (x1, y1, x2, y2), tid in zip(boxes, ids):
                    hist = history[tid]
                    if hist and frame_idx - hist[-1][0] > max_gap:   # track was lost for too long
                        hist.clear()
                        above[tid] = 0
                    hist.append((frame_idx, (x1 + x2) / 2, y2, max(y2 - y1, 1)))

                    # speed is only evaluated once the window is full
                    speed = measure_speed(hist, fps, unit, person_height_m) \
                        if len(hist) == hist.maxlen else None

                    if speed is not None:
                        if speed > threshold:
                            above[tid] += 1
                            if above[tid] >= confirm_frames:
                                fast_ids.add(tid)                    # latch: never removed
                        else:
                            above[tid] = 0

                    color = GREEN if tid in fast_ids else YELLOW
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"#{tid}" if speed is None else f"#{tid} {speed:.1f} {unit_label}"
                    draw_label(frame, x1, y1, label, color)

            cv2.putText(frame, f"now: {n_now}  fast IDs: {len(fast_ids)}  (> {threshold} {unit_label})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return {"out_path": out_path, "fast_ids": sorted(fast_ids)}




