"""Multivariate LSTM forecaster (from the Dhwani heat/sound/power node).

One LSTM: all channels in, all channels out. Lookback window -> next step,
rolled forward autoregressively for `horizon` steps. Seeded at 42, MinMax-scaled.

Standalone version: the two Dhwani imports (CHANNELS, extract) are replaced by a
local CHANNELS list. Edit CHANNELS to match your streams.

Input format everywhere: a "cols" dict per window, e.g.
    {"temp_c": [..], "hr_bpm": [..], "spo2": [..]}
"""
import json
import os
import random
from pathlib import Path

import numpy as np

CHANNELS = ["temp_c", "hr_bpm", "spo2"]   # placeholder: set to your channels

# absolute minimum band half-width per channel: stops a near-constant (filled/unused)
# channel from producing a divide-by-tiny blow-up. Values = smallest move worth flagging.
FLOOR = {"temp_c": 0.4, "hr_bpm": 6.0, "spo2": 1.5}

SEED = 42
LOOKBACK = 12
LSTM_UNITS = 48
EPOCHS = 40
BATCH = 16


def _seed_everything():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    import tensorflow as tf
    tf.random.set_seed(SEED)


def _stack_raw(cols_list):
    """List of per-window column dicts -> one (T, n_channels) array."""
    rows = []
    for cols in cols_list:
        n = len(cols[CHANNELS[0]])
        for i in range(n):
            rows.append([cols[ch][i] for ch in CHANNELS])
    return np.asarray(rows, dtype=float)


class HSPForecaster:
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler

    # ---- training -------------------------------------------------------
    @classmethod
    def fit(cls, healthy_cols_list, lookback=LOOKBACK, epochs=EPOCHS):
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Input
        _seed_everything()

        raw = _stack_raw(healthy_cols_list)
        scaler = MinMaxScaler().fit(raw)
        scaled = scaler.transform(raw)

        X, y = [], []
        for i in range(len(scaled) - lookback):
            X.append(scaled[i:i + lookback])
            y.append(scaled[i + lookback])
        X, y = np.asarray(X), np.asarray(y)
        if len(X) < 8:
            raise ValueError(f"not enough data to train (got {len(X)} windows); "
                             f"capture more baseline")

        model = Sequential([
            Input(shape=(lookback, len(CHANNELS))),
            LSTM(LSTM_UNITS),
            Dense(len(CHANNELS)),
        ])
        model.compile(optimizer="adam", loss="mae")
        model.fit(X, y, epochs=epochs, batch_size=BATCH, verbose=0)
        return cls(model=model, scaler=scaler)

    # ---- forecasting ----------------------------------------------------
    def _fast_step(self, window):
        """One-step prediction via a cached tf.function (avoids model.predict overhead,
        which dominates when called in a rollout loop). Falls back to __call__."""
        import tensorflow as tf
        if getattr(self, "_stepfn", None) is None:
            self._stepfn = tf.function(
                lambda x: self.model(x, training=False),
                input_signature=[tf.TensorSpec([1, window.shape[0], window.shape[1]], tf.float32)])
        return self._stepfn(tf.constant(window[np.newaxis].astype("float32"))).numpy()[0]

    def forecast(self, recent_cols_list, horizon):
        """Autoregressive rollout. Returns {channel: [values]} in raw units."""
        raw = _stack_raw(recent_cols_list)
        lookback = self.model.input_shape[1]
        if len(raw) < lookback:
            raise ValueError(f"need >= {lookback} recent samples, got {len(raw)}")
        window = self.scaler.transform(raw[-lookback:])
        preds = []
        for _ in range(horizon):
            nxt = self._fast_step(window)                 # cached graph, no per-call predict overhead
            preds.append(nxt)
            window = np.vstack([window[1:], nxt])
        preds = self.scaler.inverse_transform(np.asarray(preds))
        return {ch: preds[:, j].tolist() for j, ch in enumerate(CHANNELS)}

    def predict_future_anomaly(self, recent_cols_list, horizon, detector=None, k=2.5):
        """Screen the forecast against an EARLY-baseline band per channel.

        Two guards make this work in the live demo:
          1. Baseline = the FIRST third of the recent buffer, not the whole window, so a
             crash already underway doesn't poison the 'normal' it's compared against.
          2. The band half-width is max(k*IQR, a per-channel absolute floor), so a channel
             held at a near-constant value (a filled/unused sensor) can't produce a
             divide-by-tiny blow-up. Floors are clinically meaningful minimums.
        `detector` is unused; kept for signature compatibility."""
        import math
        forecast_cols = self.forecast(recent_cols_list, horizon)
        recent_raw = _stack_raw(recent_cols_list)
        base_n = max(5, len(recent_raw) // 3)              # early slice = the 'normal' reference

        worst = {"channel": None, "z": 0.0, "dir": "", "at": None}
        details = {}
        for j, ch in enumerate(CHANNELS):
            base = recent_raw[:base_n, j]
            q1, med, q3 = np.percentile(base, [25, 50, 75])
            half = max((q3 - q1) * (k / 1.349), FLOOR.get(ch, 1.0))   # IQR->~std, floored
            upper, lower = med + k * half, med - k * half
            fut = np.asarray(forecast_cols[ch])
            over = np.maximum(0.0, (fut - upper) / half)
            under = np.maximum(0.0, (lower - fut) / half)
            z_series = np.maximum(over, under)
            zmax = float(z_series.max())
            first = int(np.argmax(z_series >= 1.0)) + 1 if (z_series >= 1.0).any() else None
            details[ch] = {"forecast_end": round(float(fut[-1]), 3),
                           "baseline_median": round(float(med), 3),
                           "sigma_beyond": round(zmax, 2)}
            if zmax > worst["z"]:
                worst = {"channel": ch, "z": zmax,
                         "dir": "rising" if over.max() >= under.max() else "falling",
                         "at": first}

        score = round(1 / (1 + math.exp(-(worst["z"] - 2.0))), 3) if worst["channel"] else 0.0
        will = bool(score >= 0.70)
        if will:
            reason = (f"{worst['channel']} forecast {worst['dir']} "
                      f"{worst['z']:.1f} sigma beyond baseline "
                      f"~{worst['at']} steps ahead")
        else:
            reason = "forecast stays within the baseline band"

        return {
            "horizon_steps": horizon,
            "will_be_anomalous": will,
            "future_anomaly_score": score,
            "first_anomalous_step": worst["at"] if will else None,
            "channel": worst["channel"] if will else None,
            "reason": reason,
            "forecast_summary": {ch: round(float(np.mean(v)), 3)
                                 for ch, v in forecast_cols.items()},
            "forecast_trend": {ch: round(float(v[-1] - v[0]), 3)
                               for ch, v in forecast_cols.items()},
            "per_channel": details,
        }

    # ---- persistence ----------------------------------------------------
    def save(self, path):
        import joblib
        stem = Path(path).with_suffix("")
        self.model.save(f"{stem}.keras")
        joblib.dump(self.scaler, f"{stem}.scaler.joblib")
        Path(f"{stem}.fc.json").write_text(json.dumps(
            {"seed": SEED, "lookback": int(self.model.input_shape[1]),
             "units": LSTM_UNITS}, indent=2))

    @classmethod
    def load(cls, path):
        import joblib
        from tensorflow.keras.models import load_model
        stem = Path(path).with_suffix("")
        model = load_model(f"{stem}.keras")
        scaler = joblib.load(f"{stem}.scaler.joblib")
        return cls(model=model, scaler=scaler)
