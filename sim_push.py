#!/usr/bin/env python3
"""Push simulated vitals to Firebase at 1 Hz -- test firebase_service.py without the ESP32.

    export FIREBASE_API_KEY='AIza...'
    export FIREBASE_DB_URL='https://....firebasedatabase.app'
    python3 sim_push.py --scenario critical          # stable | deteriorating | critical | glitch

Writes  ambulink/amb/{pid}/vitals/{pushId}  {"t", "temp_c", "hr_bpm", "spo2"}
"""
import argparse
import os
import time

from firebase_rest import ApiKeyDB
from vitals_sim import make_stream

CHANNELS = ["temp_c", "hr_bpm", "spo2"]

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="critical",
                choices=["stable", "deteriorating", "critical", "glitch"])
ap.add_argument("--pid", default="P001")
ap.add_argument("--n", type=int, default=240, help="seconds of data to send")
ap.add_argument("--onset", type=int, default=60, help="second the problem starts")
ap.add_argument("--rate", type=float, default=1.0, help="readings per second")
ap.add_argument("--temp-only", action="store_true", help="send only temp_c (like the Classic)")
ap.add_argument("--api-key", default=os.environ.get("FIREBASE_API_KEY"))
ap.add_argument("--db-url", default=os.environ.get("FIREBASE_DB_URL"))
a = ap.parse_args()
if not a.api_key or not a.db_url:
    ap.error("set FIREBASE_API_KEY and FIREBASE_DB_URL (or pass --api-key / --db-url)")

db = ApiKeyDB(a.api_key, a.db_url)
ref = db.reference(f"ambulink/amb/{a.pid}/vitals")
data = make_stream(a.scenario, n=a.n, onset=a.onset, seed=int(time.time()) % 1000)
print(f"pushing {a.n} readings for {a.pid} ({a.scenario}, problem at {a.onset}s) ... Ctrl+C to stop")
try:
    for i, row in enumerate(data):
        sample = {"t": int(time.time())}
        for j, ch in enumerate(CHANNELS):
            if a.temp_only and ch != "temp_c":
                continue
            sample[ch] = round(float(row[j]), 2)
        try:
            ref.push(sample)
        except Exception as e:                          # network gave up: skip this reading
            print(f"{i + 1:4d}  skipped ({type(e).__name__}) -- check Wi-Fi")
            continue
        print(f"{i + 1:4d}  " + "  ".join(f"{k}={v}" for k, v in sample.items() if k != "t"))
        time.sleep(1 / a.rate)
except KeyboardInterrupt:
    print("stopped")
