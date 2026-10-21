"""Offline test of firebase_service (no Firebase, no audio): uses the shipped lstm_model.*

Feeds fake Firebase listener events through the real listener callback -> queue -> worker path.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings; warnings.filterwarnings("ignore")
import queue
import threading
import time
from types import SimpleNamespace

from forecaster import HSPForecaster
from vitals_sim import make_stream
import firebase_service as fs

fc = HSPForecaster.load("lstm_model")
writes, deletes, said = [], [], []
pipe = fs.Pipeline(fc, lambda p, d: writes.append((p, d)), said.append,
                   delete=deletes.append, horizon=15, recent=45, step=5, repeat_s=2, stale_s=3)

# ---- parser unit checks ------------------------------------------------------------
s = {"t": 1, "temp_c": 36.8}
assert fs.extract_samples(["P1", "vitals", "-Nabc"], s) == [("P1", "-Nabc", s)]          # push child
assert fs.extract_samples(["P1", "vitals"], {"-N1": s, "-N2": s})[1][1] == "-N2"          # batch
assert fs.extract_samples(["P1", "vitals", "latest"], s)[0][1] == "latest"                # overwrite
assert fs.extract_samples(["P1"], {"vitals": {"-N1": s}, "name": "x"})[0][0] == "P1"      # whole patient
assert fs.extract_samples([], {"P1": {"vitals": {"-N1": s}}, "P2": {"vitals": {"-N2": s}}})[1][0] == "P2"
assert fs.extract_samples(["P1", "vitals", "-N1", "temp_c"], 36.8) == []                 # scalar field: ignored
assert fs.extract_samples(["P1", "forecast"], {"temp_c": 1}) == []                        # not under vitals
assert fs.extract_samples(["P1", "vitals"], [None, s, s])[0][1] == "1"                    # firebase list form
assert fs.command_pids(["P1"], True) == ["P1"] and fs.command_pids(["P1"], None) == []
assert fs.to_msg({"temp_c": "37.5", "hr_bpm": None, "spo2": float("nan")}) == {"temp_c": 37.5}
assert fs.spell("P001") == "P 0 0 1"
print("parser checks ok")

# ---- end-to-end via the real listener callback + worker thread ---------------------
inq, stop = queue.Queue(), threading.Event()
th = threading.Thread(target=fs.worker, args=(inq, pipe, stop), daemon=True); th.start()
vcb = fs.make_listener(inq, "vitals")
hcb = fs.make_listener(inq, "handoff")
ccb = fs.make_listener(inq, "clear")
ev = lambda t, path, data: SimpleNamespace(event_type=t, path=path, data=data)

vcb(ev("put", "/", {"P000": {"vitals": {"-Nold%d" % i: {"t": i, "temp_c": 50} for i in range(500)}}}))  # initial snapshot
hcb(ev("put", "/", None)); ccb(ev("put", "/", None))

N = 90
crit = make_stream("critical", n=N, onset=40, seed=101)
stab = make_stream("stable", n=N, seed=202)
for i in range(N):
    row = lambda a, full=True: ({"t": i, "temp_c": float(a[i, 0]), "hr_bpm": float(a[i, 1]), "spo2": float(a[i, 2])}
                                if full else {"t": i, "temp_c": float(a[i, 0])})
    vcb(ev("put", f"/P001/vitals/-N{i:05d}", row(crit)))                 # push-id style
    vcb(ev("put", f"/P002/vitals/-N{i:05d}", row(stab, full=False)))     # temp-only patient
    if i == 10:
        vcb(ev("put", f"/P001/vitals/-N{i:05d}", row(crit)))             # duplicate re-delivery
while not inq.empty():
    time.sleep(0.05)
time.sleep(0.3)

p1_alerts = [t for t in said if "Patient P 0 0 1" in t and ("Warning" in t or "Critical" in t)]
p2_alerts = [t for t in said if "P 0 0 2" in t]
p1_fc = [d for p, d in writes if p == "ambulink/forecast/P001" and d.get("ready")]
print("P001 voice alerts:", p1_alerts)
print("P002 voice (should be none):", p2_alerts)
print("forecast writes P001:", len(p1_fc), "| any P000 buffered from snapshot:", "P000" in pipe.svc.pats)
assert p1_alerts, "crash not announced"
assert not p2_alerts, "stable patient false-alarmed"
assert "P000" not in pipe.svc.pats, "initial snapshot was ingested as live data"
assert len(pipe.svc.pats["P001"].raw) == N, "duplicate not de-duplicated"
assert all(isinstance(d.get("will_be_anomalous"), bool) for d in p1_fc)

# ---- reminders while latched --------------------------------------------------------
n = len(said); time.sleep(3.2)
assert any("Reminder" in t for t in said[n:]), "no reminder while latched"
print("reminder ok:", [t for t in said[n:] if "Reminder" in t][0])

# ---- stale watchdog + recovery -----------------------------------------------------
assert any("No vitals received from patient P 0 0 2" in t for t in said), "no stale warning"
vcb(ev("put", "/P002/vitals/-Nback", {"t": 999, "temp_c": 36.9}))
time.sleep(0.5)
assert any("restored" in t for t in said), "no restored message"
print("stale/restore ok")

# ---- clear, handoff ----------------------------------------------------------------
ccb(ev("put", "/P001", True)); time.sleep(0.5)
assert "ambulink/clear/P001" in deletes and pipe.level.get("P001") is None
assert len(pipe.svc.pats["P001"].raw) == 0
hcb(ev("put", "/P002", "go")); time.sleep(0.5)
assert any(p == "ambulink/icu/P002/handoff" for p, _ in writes), "handoff not written"
assert "ambulink/handoff/P002" in deletes
n = len(writes)
vcb(ev("put", "/P002/vitals/-Nafter", {"t": 1000, "temp_c": 37.0})); time.sleep(0.3)
assert len(writes) == n, "vitals after handoff not ignored"
hcb(ev("put", "/P999", True)); time.sleep(0.3)          # unknown id must not crash the worker
print("clear + handoff ok")

lat = [d["latency_ms"] for p, d in writes if p == "ambulink/forecast/P001" and "latency_ms" in d]
print(f"forecast latency on this box: median {sorted(lat)[len(lat)//2]} ms")
stop.set()
print("FIREBASE SERVICE TEST PASSED")
