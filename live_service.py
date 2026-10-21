"""AMBULINK LSTM live service: ESP32 vitals over MQTT -> rolling buffer -> forecaster.

Flow
  ESP32 (Classic temp over UART + any onboard HR/SpO2) -> Wi-Fi -> MQTT
  this service subscribes, keeps a per-patient rolling buffer, and every
  `step` seconds runs forecast + future-anomaly screening, then publishes the
  verdict and speaks an alert when a NEW anomaly appears.

Topics (pid = P001 etc.)
  in : ambulink/amb/{pid}/vitals      {"t":..,"temp_c":36.8,"hr_bpm":80,"spo2":97}
  in : ambulink/handoff/{pid}         any payload (ER -> ICU handoff)
  out: ambulink/amb/{pid}/forecast    verdict JSON (dashboard subscribes)
  out: ambulink/icu/{pid}/handoff     recent history + last verdict

Missing channels (e.g. only temp is wired) are held at a configurable resting
value so the live demo runs with just the Circuit Playground thermistor.

Run:  python3 live_service.py --model lstm_model --broker 127.0.0.1
      (train + save a model first: see README)
"""
import argparse
import json
import subprocess
import time
from collections import deque

import numpy as np

from forecaster import HSPForecaster, CHANNELS, LOOKBACK

# resting fill for channels the hardware doesn't provide yet
RESTING = {"temp_c": 36.8, "hr_bpm": 78, "spo2": 98}


class PatientBuffer:
    """Rolling raw-sample buffer for one patient, as the forecaster's cols format."""
    def __init__(self, keep):
        self.raw = deque(maxlen=keep)       # list of [temp, hr, spo2]
        self.last_verdict = None
        self.last_anom = False
        self.n = 0                          # readings seen (for the forecast interval)

    def add(self, sample):
        self.raw.append(sample)

    def recent_cols(self, n):
        """Last n samples as a single-window cols dict the forecaster accepts."""
        arr = np.asarray(list(self.raw)[-n:], dtype=float)
        return [{ch: arr[:, j].tolist() for j, ch in enumerate(CHANNELS)}]


class LiveForecastService:
    def __init__(self, fc, publish, speak=print, horizon=20, history=120,
                 recent=30, step=5, fill_missing=True):
        self.fc, self.publish, self.speak = fc, publish, speak
        self.horizon, self.recent, self.fill = horizon, recent, fill_missing
        self.step = step                              # run a forecast every `step` readings
        self.lookback = fc.model.input_shape[1]
        self.need = max(self.lookback, recent)        # samples before first forecast
        self.history = max(history, self.need)
        self.pats = {}
        self.handed_off = set()

    def _buf(self, pid):
        if pid not in self.pats:
            self.pats[pid] = PatientBuffer(self.history)
        return self.pats[pid]

    def on_vitals(self, pid, msg):
        if pid in self.handed_off:
            return None
        buf = self._buf(pid)
        row = []
        for ch in CHANNELS:
            v = msg.get(ch)
            if v is None:
                if not self.fill:
                    return None                       # incomplete reading, skip
                v = RESTING[ch]
            row.append(float(v))
        buf.add(row)
        buf.n += 1

        if len(buf.raw) < self.need:                  # still filling the window
            out = {"ready": False, "patient_id": pid, "have": len(buf.raw),
                   "need": self.need, "state": "warming up"}
            self.publish(f"ambulink/amb/{pid}/forecast", out)
            return out

        # only run the (costly) forecast every `step` readings; echo last verdict otherwise
        if buf.last_verdict is not None and (buf.n % self.step) != 0:
            return buf.last_verdict

        t0 = time.perf_counter()
        v = self.fc.predict_future_anomaly(buf.recent_cols(self.recent), self.horizon)
        v.update({"ready": True, "patient_id": pid,
                  "vitals": dict(zip(CHANNELS, row)),
                  "ts": msg.get("t", time.time()),
                  "latency_ms": round((time.perf_counter() - t0) * 1000, 2)})
        # latch: once an anomaly is raised it stays raised (a clinical alert shouldn't
        # silently clear just because the crashed signal has plateaued). Clinician clears it.
        if v["will_be_anomalous"] and not buf.last_anom:
            ahead = v.get("first_anomalous_step")
            when = f"in about {ahead} seconds" if ahead else "soon"
            self.speak(f"Warning. Patient {pid}. {v['channel']} predicted abnormal {when}. {v['reason']}.")
            buf.last_anom = True
        if buf.last_anom and not v["will_be_anomalous"]:
            v["will_be_anomalous"] = True
            v["latched"] = True
        buf.last_verdict = v

        self.publish(f"ambulink/amb/{pid}/forecast", v)
        return v

    def on_handoff(self, pid):
        if pid not in self.pats:
            self.publish("ambulink/handoff/errors", {"patient_id": pid, "error": "unknown patient_id"})
            return None
        if pid in self.handed_off:
            return None
        self.handed_off.add(pid)
        buf = self.pats[pid]
        payload = {"patient_id": pid, "last_verdict": buf.last_verdict,
                   "history": [dict(zip(CHANNELS, r)) for r in buf.raw],
                   "handoff_ts": time.time()}
        self.publish(f"ambulink/icu/{pid}/handoff", payload)
        self.speak(f"Patient {pid} incoming to ICU.")
        return payload


def espeak(text):
    try:
        subprocess.Popen(["espeak-ng", "-s", "150", text])
    except FileNotFoundError:
        print("[voice]", text)


def main():
    import paho.mqtt.client as mqtt
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm_model")
    ap.add_argument("--broker", default="127.0.0.1")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--step", type=int, default=5, help="run a forecast every N readings")
    ap.add_argument("--no-fill", action="store_true", help="require all channels, don't fill missing")
    a = ap.parse_args()

    fc = HSPForecaster.load(a.model)
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()
    svc = LiveForecastService(fc, lambda t, p: client.publish(t, json.dumps(p)),
                              speak=espeak, horizon=a.horizon, step=a.step,
                              fill_missing=not a.no_fill)

    def on_connect(c, *args):
        c.subscribe([("ambulink/amb/+/vitals", 0), ("ambulink/handoff/+", 0)])
        print(f"connected; need {svc.need} samples before first forecast")

    def on_message(c, _u, m):
        parts = m.topic.split("/")
        try:
            if parts[1] == "amb" and parts[3] == "vitals":
                svc.on_vitals(parts[2], json.loads(m.payload.decode()))
            elif parts[1] == "handoff":
                svc.on_handoff(parts[2])
        except Exception as e:
            print("bad message", m.topic, repr(e)[:120])

    client.on_connect, client.on_message = on_connect, on_message
    client.connect(a.broker, 1883, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()
