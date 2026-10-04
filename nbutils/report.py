"""Analysis of a finished session, for notebook cells after the live run."""
import matplotlib.pyplot as plt
import numpy as np


def summary(est):
    raw = np.array(est.raw)
    if len(raw) < 2:
        return {"error": "no samples (no face was detected)"}
    dur = raw[-1, 0] - raw[0, 0]
    out = {"samples": len(raw), "duration_s": round(float(dur), 1),
           "sample_rate_hz": round(float((len(raw) - 1) / dur), 1)}
    if est.history:
        h = np.array(est.history)                 # t, bpm_instant, bpm_smoothed, confidence
        good = h[:, 3] >= est.conf_ok
        out["estimates"] = len(h)
        out["confident_share"] = round(float(good.mean()), 2)
        out["median_bpm_all"] = round(float(np.median(h[:, 2])), 1)
        if good.any():
            out["median_bpm_confident"] = round(float(np.median(h[good, 2])), 1)
            out["std_bpm_confident"] = round(float(np.std(h[good, 2])), 1)
    full = est.estimate_full()
    if full is not None:
        out["full_session_bpm"] = round(float(full.bpm), 1)
        out["full_session_confidence"] = round(float(full.snr), 2)
    return out


def plot_session(est, **full_kwargs):
    """3 plots: signal over the session, BPM over time, whole-session spectrum.
    full_kwargs (fmin, fmax, channels) are passed to est.estimate_full()."""
    raw = np.array(est.raw)
    if len(raw) < 10:
        print("Not enough data to plot.")
        return
    t = raw[:, 0] - raw[0, 0]
    fs = (len(raw) - 1) / t[-1]
    k = max(int(2 * fs), 3)                                   # ~2 s moving average as a high-pass
    g = raw[:, 2] - np.convolve(raw[:, 2], np.ones(k) / k, mode="same")

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(t, g, lw=0.8, color="tab:green")
    ax[0].set(title="Forehead green channel (high-passed)", xlabel="t [s]")

    if est.history:
        h = np.array(est.history)
        ht = h[:, 0] - raw[0, 0]
        ax[1].plot(ht, h[:, 2], color="k", lw=1, label="smoothed")
        sc = ax[1].scatter(ht, h[:, 1], c=h[:, 3], cmap="viridis", s=12, label="instant (colour = confidence)")
        plt.colorbar(sc, ax=ax[1], label="confidence")
        ax[1].legend(loc="upper right")
    ax[1].set(title="BPM over time", xlabel="t [s]", ylabel="bpm")

    full = est.estimate_full(**full_kwargs)
    if full is not None:
        ax[2].plot(full.bpm_axis, full.power / full.power.max())
        ax[2].axvline(full.bpm, color="r", ls="--", label=f"{full.bpm:.1f} bpm (conf {full.snr:.2f})")
        ax[2].legend()
    ax[2].set(title="Whole-session spectrum", xlabel="bpm")
    plt.tight_layout()
    plt.show()