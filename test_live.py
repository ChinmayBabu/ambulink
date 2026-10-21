"""Offline test of live_service.LiveForecastService (no broker): ESP32-style feed, isolation, handoff."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from forecaster import HSPForecaster, CHANNELS
from vitals_sim import make_stream
from live_service import LiveForecastService


def cols(arr):
    return {ch: arr[:, j].tolist() for j, ch in enumerate(CHANNELS)}


fc = HSPForecaster.fit([cols(make_stream("stable", n=300, seed=s)) for s in range(30)], epochs=40)
fc.save("lstm_model")

pub, said = [], []
svc = LiveForecastService(fc, lambda t, p: pub.append((t, p)), speak=said.append,
                          horizon=15, recent=45, step=5)

# simulate ESP32 publishing one reading per second: P001 crashes, P002 stable.
# ESP32 only sends temp_c + hr_bpm + spo2 here; to mimic "temp only" hardware, drop hr/spo2 on P002.
N = 90
crit = make_stream("critical", n=N, onset=40, seed=101)
stab = make_stream("stable", n=N, seed=202)
first_alert = None
import time as _t; _wall = _t.perf_counter()
for i in range(N):
    v1 = svc.on_vitals("P001", {"t": i, "temp_c": crit[i, 0], "hr_bpm": crit[i, 1], "spo2": crit[i, 2]})
    svc.on_vitals("P002", {"t": i, "temp_c": stab[i, 0]})      # temp-only -> hr/spo2 filled
    if v1 and v1.get("ready") and v1["will_be_anomalous"] and first_alert is None:
        first_alert = i

v1 = svc.pats["P001"].last_verdict
v2 = svc.pats["P002"].last_verdict
p2_ever = any(p.get("will_be_anomalous") for t, p in pub if t == "ambulink/amb/P002/forecast")
print(f"ran {N} readings x 2 patients in {_t.perf_counter()-_wall:.1f}s wall")
print("P001 first alert at t=", first_alert, "(onset 40), latched state now:", v1["will_be_anomalous"], "channel:", v1["channel"])
print("P002 ever flagged:", p2_ever, "(temp-only, filled hr/spo2 -> should be False)")
print("latency per forecast (this box):", v1["latency_ms"], "ms")
print("voice alerts:", said[:3])
assert first_alert is not None and first_alert >= 40, "P001 crash not detected after onset"
assert not p2_ever, "P002 (stable, filled channels) false-alarmed"

# handoff: unknown, real, duplicate, feed-after-handoff
n0 = len(pub)
svc.on_handoff("P999")
svc.on_handoff("P001"); svc.on_handoff("P001")
hand = [p for t, p in pub[n0:] if t == "ambulink/icu/P001/handoff"]
errs = [p for t, p in pub[n0:] if t == "ambulink/handoff/errors"]
print(f"handoff: {len(hand)} sent (expect 1), {len(errs)} unknown-id error (expect 1), history len {len(hand[0]['history'])}")
assert len(hand) == 1 and len(errs) == 1
assert svc.on_vitals("P001", {"t": 999, "temp_c": 37.0}) is None
print("feed after handoff ignored: True")
print("LIVE SERVICE TEST PASSED")
