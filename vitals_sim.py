"""Simulated 1 Hz vitals for demo/testing. Channels: temp_c, hr_bpm, spo2.

Scenarios: stable | deteriorating (slow drift) | critical (fast crash) | glitch (one-off spike)
Each patient gets their own baseline, so the detector is tested on unseen patients.
"""
import numpy as np


def _ar_noise(rng, n, sigma, phi=0.8):
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(0, sigma)
    return e


def make_stream(scenario="stable", n=300, onset=120, seed=0):
    """Returns (n, 3) array. `onset` = second the problem begins."""
    rng = np.random.default_rng(seed)
    t0 = rng.normal(36.8, 0.2)
    hr0 = rng.uniform(62, 95)
    sp0 = rng.uniform(96, 99)
    temp = t0 + _ar_noise(rng, n, 0.03)
    hr = hr0 + _ar_noise(rng, n, 1.2)
    spo2 = sp0 + _ar_noise(rng, n, 0.25)

    k = np.clip(np.arange(n) - onset, 0, None).astype(float)
    if scenario == "deteriorating":      # slow drift over ~2.5 min
        r = np.clip(k / 150, 0, 1)
        hr += 38 * r; spo2 -= 7 * r; temp += 1.1 * r
    elif scenario == "critical":         # fast crash over ~40 s
        r = np.clip(k / 40, 0, 1)
        hr += 65 * r; spo2 -= 13 * r; temp += 2.0 * r
    elif scenario == "glitch":           # 2-second sensor spike, then normal
        spo2[onset:onset + 2] = 70
        hr[onset:onset + 2] += 60
    return np.stack([temp, hr, spo2], axis=1)
