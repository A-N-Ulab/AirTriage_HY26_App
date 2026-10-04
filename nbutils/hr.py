"""Core heart-rate-from-face-video logic (no UI, no notebook dependencies)."""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
CHANNELS = {"g": [2], "rg": [1, 2]}      # column indices in the (t, R, G, B) samples


@dataclass
class Estimate:
    bpm: float              # frequency of the strongest peak inside the band
    bpm_axis: np.ndarray    # spectrum bins inside the band, in bpm
    power: np.ndarray       # spectrum power at those bins
    snr: float              # share of band power around the peak (0..1) = "confidence"
    t: np.ndarray           # uniform time grid of the analysed signal (s)
    signal: np.ndarray      # detrended, normalised signal that went into the FFT


def detect_face(frame_bgr, width=320):
    """Largest face as float [x, y, w, h] in full-frame pixels, or None."""
    scale = width / frame_bgr.shape[1]
    small = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
    faces = _CASCADE.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return np.array([x, y, w, h], dtype=float) / scale


def forehead_roi(face, frame_w, frame_h):
    """Forehead rectangle (x1, y1, x2, y2) derived from the face box, or None if too small."""
    x, y, w, h = face
    x1, y1 = int(max(x + 0.30 * w, 0)), int(max(y + 0.08 * h, 0))
    x2, y2 = int(min(x + 0.70 * w, frame_w)), int(min(y + 0.25 * h, frame_h))
    return (x1, y1, x2, y2) if x2 - x1 > 4 and y2 - y1 > 4 else None


def fft_bpm(ts, vals, fmin=0.7, fmax=3.0, fs=30.0, nfft=8192):
    """
    ts:   (N,) timestamps in seconds (webcam fps is not constant, so use real times)
    vals: (N, C) mean channel intensities
    """
    ts = np.asarray(ts, float)
    vals = np.asarray(vals, float)
    if len(ts) < 10 or ts[-1] - ts[0] < 1.0:
        return None

    grid = np.arange(ts[0], ts[-1], 1.0 / fs)              # uniform time grid (FFT needs it)
    g = grid - grid[0]
    sig = np.zeros(len(grid))
    for c in range(vals.shape[1]):
        x = np.interp(grid, ts, vals[:, c])
        x = x - np.polyval(np.polyfit(g, x, 1), g)         # remove slow drift
        sig += x / (x.std() + 1e-9)                        # channels weigh equally

    nfft = max(nfft, 1 << int(np.ceil(np.log2(len(sig)))))
    spec = np.abs(np.fft.rfft(sig * np.hamming(len(sig)), n=nfft)) ** 2   # zero-padded FFT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    f, p = freqs[band], spec[band]
    peak = f[np.argmax(p)]
    snr = p[np.abs(f - peak) <= 0.1].sum() / (p.sum() + 1e-12)
    return Estimate(peak * 60.0, f * 60.0, p, snr, grid, sig)


class HeartRateEstimator:
    """Feed it frames + timestamps with process(); read the results from its attributes."""

    def __init__(self, window_s=10.0, min_s=5.0, fmin=0.7, fmax=3.0, channels="g",
                 update_every=0.5, detect_every=3, conf_ok=0.2):
        self.window_s, self.min_s = window_s, min_s
        self.fmin, self.fmax = fmin, fmax
        self.cols = CHANNELS[channels]
        self.update_every, self.detect_every, self.conf_ok = update_every, detect_every, conf_ok

        self.buf = deque()            # sliding window of (t, R, G, B)
        self.raw = []                 # whole session of (t, R, G, B), for analysis after the run
        self.history = []             # (t, bpm_instant, bpm_smoothed, confidence)
        self.recent = deque(maxlen=5)
        self.face = None
        self.roi = None
        self.bpm = None               # smoothed value (median of the last estimates)
        self.est = None               # last Estimate (spectrum, signal, ...)
        self.missed = 0
        self.n = 0
        self.last_est = -1e9

    # ---- state -----------------------------------------------------------
    @property
    def status(self):
        if self.face is None:
            return "no_face"
        return "collecting" if self.bpm is None else "ok"

    @property
    def progress(self):
        return 0.0 if len(self.buf) < 2 else min((self.buf[-1][0] - self.buf[0][0]) / self.min_s, 1.0)

    def _reset_window(self):
        self.face = self.roi = self.bpm = self.est = None
        self.buf.clear()
        self.recent.clear()

    def trace(self):
        """(t, y): the current window of the first channel, detrended (for live plotting)."""
        if len(self.buf) < 5:
            return None
        arr = np.array(self.buf)
        t, y = arr[:, 0], arr[:, self.cols[0]]
        y = y - np.polyval(np.polyfit(t - t[0], y, 1), t - t[0])
        return t, y

    # ---- main entry point -----------------------------------------------
    def process(self, frame, t):
        H, W = frame.shape[:2]

        if self.n % self.detect_every == 0:                 # detect on every Nth frame only
            new = detect_face(frame)
            if new is not None:
                self.face = new if self.face is None else 0.6 * self.face + 0.4 * new  # less ROI jitter
                self.missed = 0
            else:
                self.missed += 1
                if self.missed > 10:                        # face lost for a while: drop the window
                    self._reset_window()
        self.n += 1

        self.roi = None
        if self.face is not None:
            self.roi = forehead_roi(self.face, W, H)
            if self.roi is not None:
                x1, y1, x2, y2 = self.roi
                b, g, r, _ = cv2.mean(frame[y1:y2, x1:x2])
                self.buf.append((t, r, g, b))
                self.raw.append((t, r, g, b))
                while self.buf[0][0] < t - self.window_s:
                    self.buf.popleft()

        if (self.roi is not None and self.buf[-1][0] - self.buf[0][0] >= self.min_s
                and t - self.last_est >= self.update_every):
            arr = np.array(self.buf)
            est = fft_bpm(arr[:, 0], arr[:, self.cols], self.fmin, self.fmax)
            if est is not None:
                self.recent.append(est.bpm)
                self.bpm = float(np.median(self.recent))
                self.est = est
                self.history.append((t, est.bpm, self.bpm, est.snr))
            self.last_est = t
        return self

    # ---- offline analysis of the whole recording -------------------------
    def estimate_full(self, fmin=None, fmax=None, channels=None):
        """FFT over the entire session; band/channels can be changed without re-recording.
        (If the face was lost mid-session, the gap is bridged by linear interpolation.)"""
        if len(self.raw) < 10:
            return None
        arr = np.array(self.raw)
        cols = CHANNELS[channels] if channels else self.cols
        return fft_bpm(arr[:, 0], arr[:, cols], fmin or self.fmin, fmax or self.fmax)