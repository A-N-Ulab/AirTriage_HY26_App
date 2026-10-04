import os
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")  # hides Qt font warnings

import subprocess, time
from pathlib import Path

import cv2
from IPython.display import display, clear_output, Image, Video


def get_device():
    """Best available accelerator: CUDA, Apple Silicon, or CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def play(frames, fps=25):
    """Show BGR frames inline as JPEGs. No browser codecs, no GUI."""
    delay = 1 / fps
    for frame in frames:
        t = time.time()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        clear_output(wait=True)
        display(Image(data=buf.tobytes()))
        time.sleep(max(0, delay - (time.time() - t)))


def preview(path, width=640):
    """Make a small WebM copy (plays in every browser) and embed it."""
    import imageio_ffmpeg
    path = Path(path)
    out = path.with_name(path.stem + "_preview.webm")
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path),
        "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "35", "-an", str(out)
    ], check=True)
    return Video(str(out), embed=True, mimetype="video/webm", width=width)