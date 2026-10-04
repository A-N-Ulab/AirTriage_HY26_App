"""Annotate videos with heart rate (HR) over time, estimated locally (no API): input dir -> output dir.

Works with VitalLens' local rPPG methods ("pos", "chrom", "g"), which need no API key.
Create the client with estimate_rolling_vitals=True to get an HR value that changes over time.
"""
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
GREEN, YELLOW, GREY, LIGHT = (0, 255, 0), (0, 255, 255), (160, 160, 160), (200, 200, 200)


# ---- reading VitalLens results --------------------------------------------------------------
def _vitals(result):
    return result.get("vitals") or result.get("vital_signs") or {}


def _series(result, name):
    """Per-frame rolling series for `name` ('heart_rate', ...) or None.

    Looks under 'vitals' first, then 'waveforms', in case the installed version puts it there.
    """
    key = f"rolling_{name}"
    for section in (_vitals(result), result.get("waveforms") or {}):
        v = section.get(key)
        if isinstance(v, dict):
            data = v.get("data", v.get("values", v.get("value")))
            if data is not None:
                arr = np.asarray(data, dtype=float).ravel()
                if arr.size > 1:
                    return arr
    return None


def _global(vitals, name):
    """Single whole-video value for `name`, or None."""
    v = vitals.get(name)
    if isinstance(v, dict) and v.get("value") is not None:
        try:
            return float(np.asarray(v["value"]).ravel()[0])
        except (TypeError, ValueError, IndexError):
            return None
    return None


def describe(result, indent=0):
    """Print keys/types/shapes of a VitalLens result (to see what your installed version returns)."""
    if isinstance(result, (list, tuple)):
        result = result[0]
    for k, v in result.items():
        pad = " " * indent
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            describe(v, indent + 2)
        else:
            shape = getattr(v, "shape", None)
            print(f"{pad}{k}: {type(v).__name__}{' ' + str(shape) if shape is not None else ''}")


def _at(series, i, n):
    """Value of a series at frame i of n (series length may differ from the frame count)."""
    v = series[min(int(i * len(series) / max(n, 1)), len(series) - 1)]
    return float(v) if np.isfinite(v) else None


# ---- drawing ----------------------------------------------------------------------------------
def _fmt(v):
    return "--" if v is None else f"{v:.0f}"


def _draw(frame, hr, hr_avg, hr_g, t):
    """Semi-transparent info box (top-left). hr_avg=True means the value is a whole-video average."""
    s = min(max(frame.shape[0] / 720.0, 0.6), 1.6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        (f"HR {_fmt(hr)} bpm" + (" (avg)" if hr_avg else ""), 1.0 * s,
         GREY if hr is None else (YELLOW if hr_avg else GREEN), 2),
    ]
    if not hr_avg and hr_g is not None:
        lines.append((f"whole video: HR {_fmt(hr_g)}", 0.55 * s, LIGHT, 1))
    lines.append((f"t = {t:5.1f} s", 0.55 * s, LIGHT, 1))

    sizes = [cv2.getTextSize(txt, font, sc, th)[0] for txt, sc, _, th in lines]
    pad = int(10 * s)
    box_w = max(w for w, _ in sizes) + 2 * pad
    box_h = sum(int(hh * 1.3) + int(hh * 0.3) for _, hh in sizes) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = pad
    for (txt, sc, col, th), (_, hh) in zip(lines, sizes):
        y += int(hh * 1.3)
        cv2.putText(frame, txt, (pad, y), font, sc, col, th, cv2.LINE_AA)
        y += int(hh * 0.3)


# ---- output encoding ---------------------------------------------------------------------------
def _finalize(raw_path, src_path, out_path):
    """Re-encode raw mp4v to H.264 (plays in browsers/Jupyter), copy audio from the source if any."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path), "-i", str(src_path),
               "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "20", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out_path)]
        try:
            subprocess.run(cmd, check=True)
            os.remove(raw_path)
            return
        except subprocess.CalledProcessError as e:
            print(f"  ffmpeg failed ({e}); keeping the raw mp4v file")
    else:
        print("  ffmpeg not found: saved with mp4v (may not play in browser/notebook)")
    os.replace(raw_path, out_path)


# ---- public API ----------------------------------------------------------------------------------
def annotate_video(vl, in_path, out_path):
    """
    Run VitalLens (local method) on one video and write a copy with HR overlaid at every moment.
    `vl` should be created with estimate_rolling_vitals=True to get a value that changes over time
    (needs >= 10 s of video; the first seconds show "--" until the first window is ready).
    Otherwise the whole-video average is shown (marked "(avg)").
    """
    in_path, out_path = Path(in_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = vl(str(in_path))
    if not results:
        raise RuntimeError("VitalLens returned no results (no face detected?)")
    result = results[0] if isinstance(results, (list, tuple)) else results
    vit = _vitals(result)
    hr_s = _series(result, "heart_rate")
    hr_g = _global(vit, "heart_rate")
    print(f"  rolling HR: {'yes' if hr_s is not None else 'no'} | whole video: HR {hr_g}")
    if hr_s is None:
        print("  no rolling HR series (video shorter than ~10 s, or estimate_rolling_vitals=False?) "
              "- overlay shows the whole-video value. Result structure:")
        describe(result)
    if hr_s is None and hr_g is None:
        print("  WARNING: no HR at all (video shorter than ~5 s?) - overlay will show '--'")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max(len(hr_s) if hr_s is not None else 0, 1)

    raw_path = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for {raw_path}")

    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            hr = _at(hr_s, i, n) if hr_s is not None else hr_g
            _draw(frame, hr, hr_s is None, hr_g, i / fps)
            writer.write(frame)
            i += 1
    finally:
        cap.release()
        writer.release()

    _finalize(raw_path, in_path, out_path)
    return {"out": str(out_path), "hr": hr_g, "rolling_hr": hr_s is not None}


def annotate_videos(vl, input_dir, output_dir, suffix="_hr", skip_existing=True):
    """Annotate every video in input_dir -> output_dir/<name><suffix>.mp4. Returns a list of summaries.

    Tip: put the method in the suffix (e.g. suffix="_hr_pos") so results from different methods
    don't get skipped as "already existing".
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in VIDEO_EXT and not p.name.startswith("."))
    if not videos:
        print(f"No videos in {input_dir.resolve()} (looking for {sorted(VIDEO_EXT)})")
        return []

    summary = []
    for p in videos:
        out = output_dir / f"{p.stem}{suffix}.mp4"
        if skip_existing and out.exists():
            print(f"{p.name}: skipped (output exists)")
            continue
        print(f"{p.name} -> {out.name}")
        try:
            info = annotate_video(vl, p, out)
            info["video"] = p.name
        except Exception as e:                                # one bad video must not stop the batch
            print(f"  FAILED: {e}")
            info = {"video": p.name, "error": str(e)}
        summary.append(info)
    return summary