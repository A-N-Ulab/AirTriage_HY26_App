"""Live notebook display: video and measurement panel side by side in one image."""
import time

import cv2
import numpy as np
from IPython.display import Image as IPyImage, display

from .hr import HeartRateEstimator, detect_face, forehead_roi

FONT, AA = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA


def _jpeg(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return IPyImage(data=buf.tobytes(), format="jpeg")


def show(img):
    """Show a BGR image in the notebook (cv2.imshow does not work in Jupyter)."""
    display(_jpeg(img))


def snapshot(source=0, warmup=10):
    """Grab one frame (skips a few frames so auto-exposure can settle)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    try:
        ok, frame = False, None
        for _ in range(warmup):
            ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError("Camera opened but returned no frame")
    return frame


def detect_preview(frame):
    """Draw the detected face and forehead ROI on a copy of the frame -> (image, face)."""
    out = frame.copy()
    face = detect_face(frame)
    if face is not None:
        x, y, w, h = face.astype(int)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 200, 0), 2)
        roi = forehead_roi(face, frame.shape[1], frame.shape[0])
        if roi is not None:
            cv2.rectangle(out, roi[:2], roi[2:], (0, 255, 0), 2)
    return out, face


def _polyline(img, x, y, rect, color, x_range=None, y_range=None):
    x0, y0, x1, y1 = rect
    cv2.rectangle(img, (x0, y0), (x1, y1), (70, 70, 70), 1)
    if len(x) < 2:
        return
    xa, xb = x_range or (x[0], x[-1])
    lo, hi = y_range or (float(np.min(y)), float(np.max(y)))
    xn = (np.asarray(x) - xa) / (xb - xa + 1e-9)
    yn = (np.asarray(y) - lo) / (hi - lo + 1e-9)
    pts = np.stack([x0 + xn * (x1 - x0), y1 - 2 - yn * (y1 - y0 - 4)], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, color, 1, AA)


def _draw_panel(h, w, est):
    p = np.full((h, w, 3), 24, np.uint8)
    grey = (200, 200, 200)
    st = est.status

    if st == "ok":
        conf = est.est.snr
        col = (0, 220, 0) if conf >= est.conf_ok else (0, 165, 255)   # orange = unsure peak
        cv2.putText(p, f"{est.bpm:.0f}", (16, int(0.20 * h)), FONT, 2.2, col, 3, AA)
        cv2.putText(p, "bpm", (150, int(0.20 * h)), FONT, 0.9, col, 2, AA)
        cv2.putText(p, f"confidence {conf:.2f}", (16, int(0.26 * h)), FONT, 0.5, grey, 1, AA)
    elif st == "collecting":
        cv2.putText(p, "Collecting...", (16, int(0.18 * h)), FONT, 1.1, (0, 255, 255), 2, AA)
        cv2.rectangle(p, (16, int(0.22 * h)), (16 + int(0.5 * w * est.progress), int(0.24 * h)),
                      (0, 255, 255), -1)
    else:
        cv2.putText(p, "No face", (16, int(0.18 * h)), FONT, 1.1, (0, 0, 255), 2, AA)

    # signal trace (current window)
    tr_rect = (10, int(0.34 * h), w - 10, int(0.60 * h))
    cv2.putText(p, "signal (detrended)", (12, tr_rect[1] - 5), FONT, 0.45, grey, 1, AA)
    tr = est.trace()
    _polyline(p, *(tr if tr else ([], [])), tr_rect, (255, 200, 0))

    # spectrum + peak marker
    sp_rect = (10, int(0.68 * h), w - 10, int(0.90 * h))
    a0, a1 = est.fmin * 60, est.fmax * 60
    cv2.putText(p, "spectrum (FFT)", (12, sp_rect[1] - 5), FONT, 0.45, grey, 1, AA)
    if est.est is not None:
        e = est.est
        _polyline(p, e.bpm_axis, e.power, sp_rect, (0, 255, 0), x_range=(a0, a1),
                  y_range=(0.0, float(e.power.max())))
        xp = int(sp_rect[0] + (e.bpm - a0) / (a1 - a0) * (sp_rect[2] - sp_rect[0]))
        cv2.line(p, (xp, sp_rect[1]), (xp, sp_rect[3]), (0, 0, 255), 1)
    else:
        cv2.rectangle(p, sp_rect[:2], sp_rect[2:], (70, 70, 70), 1)
    for b in (60, 90, 120, 150, 180):
        if a0 <= b <= a1:
            xt = int(sp_rect[0] + (b - a0) / (a1 - a0) * (sp_rect[2] - sp_rect[0]))
            cv2.line(p, (xt, sp_rect[3]), (xt, sp_rect[3] + 4), grey, 1)
            cv2.putText(p, str(b), (xt - 10, int(0.96 * h)), FONT, 0.4, grey, 1, AA)
    return p


def compose(frame, est, display_h=360, panel_w=440):
    """Video (with face + forehead boxes) on the left, measurement panel on the right."""
    vis = frame.copy()
    if est.face is not None:
        x, y, w, h = est.face.astype(int)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 200, 0), 2)
    if est.roi is not None:
        cv2.rectangle(vis, est.roi[:2], est.roi[2:], (0, 255, 0), 2)
    s = display_h / vis.shape[0]
    vis = cv2.resize(vis, (int(vis.shape[1] * s), display_h))
    return np.hstack([vis, _draw_panel(display_h, panel_w, est)])


def run_live(source=0, duration=30, estimator=None, display_h=360, max_display_fps=20, **est_kwargs):
    """
    Live preview inside the notebook cell. Stops after `duration` seconds (None = until you press
    the interrupt button ■, which is handled gracefully). Returns the estimator with the whole
    session in it (raw, history, ...), so later cells can analyse it.
    """
    est = estimator or HeartRateEstimator(**est_kwargs)
    src = int(source) if isinstance(source, str) and source.isdigit() else source
    is_file = isinstance(src, str)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    handle, last_show, n, t0 = None, 0.0, 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = n / file_fps if is_file else time.perf_counter() - t0
            if duration and t >= duration:
                break
            est.process(frame, t)
            n += 1

            now = time.perf_counter()
            if now - last_show >= 1.0 / max_display_fps:     # throttle the browser updates
                img = _jpeg(compose(frame, est, display_h))
                if handle is None:
                    handle = display(img, display_id=True)
                else:
                    handle.update(img)
                last_show = now
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()                                         # always free the camera
    return est