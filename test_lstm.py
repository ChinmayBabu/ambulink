"""Train the LSTM forecaster on simulated healthy vitals, then check forecasts on healthy vs degrading patients."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from forecaster import HSPForecaster, CHANNELS
from vitals_sim import make_stream


def to_cols(arr):                       # (T,3) -> {"temp_c": [...], ...} as forecaster expects
    return {ch: arr[:, j].tolist() for j, ch in enumerate(CHANNELS)}


train = [to_cols(make_stream("stable", n=300, seed=s)) for s in range(30)]
fc = HSPForecaster.fit(train, epochs=40)
print("trained")

for scen in ("stable", "deteriorating", "critical"):
    hits, flagged = 0, []
    for s in range(100, 110):
        st = make_stream(scen, n=300, onset=60, seed=s)
        recent = [to_cols(st[100:130])]           # 30 s of recent history, after onset
        v = fc.predict_future_anomaly(recent, horizon=20)
        hits += v["will_be_anomalous"]
        flagged.append(v["channel"])
    print(f"{scen:13s}: forecast flags anomaly in {hits}/10 patients | channels: {sorted(set(map(str, flagged)))}")
