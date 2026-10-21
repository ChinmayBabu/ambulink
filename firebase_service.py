#!/usr/bin/env python3
"""AMBULINK on Raspberry Pi 5: Firebase vitals -> LSTM forecast -> voice alerts.

Firebase Realtime Database layout (all paths configurable below)

  IN   ambulink/amb/{pid}/vitals/{pushId}   {"t": 123, "temp_c": 37.1, "hr_bpm": 80, "spo2": 97}
       ambulink/handoff/{pid}               write anything (e.g. true) -> ER->ICU handoff
       ambulink/clear/{pid}                 write anything (e.g. true) -> clinician clears latched alert
  OUT  ambulink/forecast/{pid}              latest verdict JSON (dashboard listens here)
       ambulink/icu/{pid}/handoff           recent history + last verdict

Command nodes (handoff / clear) are deleted after they are processed so they can be re-sent.

Run (API key -- anonymous sign-in; enable Anonymous in Firebase Authentication)
    python3 firebase_service.py --model lstm_model \
        --api-key AIza... \
        --db-url https://YOUR-PROJECT-default-rtdb.firebaseio.com

Run (service account -- full admin access, rules don't apply)
    python3 firebase_service.py --model lstm_model \
        --cred serviceAccountKey.json \
        --db-url https://YOUR-PROJECT-default-rtdb.firebaseio.com

Threads
    Firebase listener thread  -> puts raw events on a queue (never blocks)
    worker thread             -> parse, buffer, LSTM forecast, decide alerts   (owns ALL state)
    speaker thread            -> speaks one message at a time
    writer thread             -> writes verdicts back to Firebase (a slow network can't stall alerts)
"""
import argparse
import json
import logging
import math
import os
import queue
import re
import signal
import subprocess
import threading
import time
from collections import deque

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")      # Pi 5 has no CUDA; skip the probe

from forecaster import HSPForecaster, CHANNELS            # noqa: E402
from live_service import LiveForecastService, RESTING     # noqa: E402
from dashboard import dashboard_view, handed_off_view      # noqa: E402

log = logging.getLogger("ambulink")

# how each channel is spoken: (name, unit, decimals)
SPOKEN = {
    "temp_c": ("temperature", "degrees celsius", 1),
    "hr_bpm": ("heart rate", "beats per minute", 0),
    "spo2": ("oxygen saturation", "percent", 0),
}
CRITICAL_SCORE = 0.90          # forecaster flags at 0.70; >= this is spoken as "Critical alert"


# --------------------------------------------------------------------------------------
# Voice
# --------------------------------------------------------------------------------------
class Speaker:
    """Serialised text-to-speech (espeak-ng). Messages never overlap.

    The Pi 5 has NO 3.5 mm jack: use a USB speaker, Bluetooth, or HDMI audio. To pick a
    specific ALSA device pass aplay_device (e.g. "plughw:2,0"; list with `aplay -l`)."""

    def __init__(self, voice="en", rate=150, amplitude=200, aplay_device=None, enabled=True):
        self.voice, self.rate, self.amp = voice, str(rate), str(amplitude)
        self.aplay_device, self.enabled = aplay_device, enabled
        self.q = queue.Queue(maxsize=20)
        self._missing_warned = False
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name="speaker")
        self._t.start()

    def say(self, text):
        log.warning("[VOICE] %s", text)
        try:
            self.q.put_nowait(text)
        except queue.Full:
            log.error("speaker queue full, dropped: %s", text)

    def _run(self):
        while not self._stop.is_set():
            try:
                text = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self.enabled:
                    self._speak(text)
            finally:
                self.q.task_done()

    def _speak(self, text):
        base = ["espeak-ng", "-v", self.voice, "-s", self.rate, "-a", self.amp]
        try:
            if self.aplay_device:
                tts = subprocess.Popen(base + ["--stdout", text], stdout=subprocess.PIPE)
                play = subprocess.Popen(["aplay", "-q", "-D", self.aplay_device], stdin=tts.stdout)
                tts.stdout.close()
                play.wait(timeout=60)
                tts.wait(timeout=5)
            else:
                subprocess.run(base + [text], timeout=60, check=False)
        except FileNotFoundError:
            if not self._missing_warned:
                log.error("espeak-ng/aplay not found: sudo apt install espeak-ng alsa-utils")
                self._missing_warned = True
        except subprocess.TimeoutExpired:
            log.error("speech timed out")

    def wait(self):
        self.q.join()

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------------------
# Firebase writer (separate thread so a slow network never blocks forecasting/alerts)
# --------------------------------------------------------------------------------------
class FirebaseWriter:
    def __init__(self, db):
        self.db = db
        self.q = queue.Queue(maxsize=500)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name="fb-writer")
        self._t.start()

    def set(self, path, payload):
        self._put(("set", path, payload))

    def delete(self, path):
        self._put(("delete", path, None))

    def _put(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:                       # drop the oldest write, keep the newest
                self.q.get_nowait()
                self.q.put_nowait(item)
            except queue.Empty:
                pass

    def _run(self):
        while not self._stop.is_set():
            try:
                op, path, payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                ref = self.db.reference(path)
                if op == "set":
                    ref.set(json.loads(json.dumps(payload, default=str)))
                else:
                    ref.delete()
            except Exception as e:     # network hiccup: log and carry on
                log.error("firebase %s %s failed: %r", op, path, e)

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------------------
# Event parsing (shape-tolerant: push-id children, overwrite style, batches, lists)
# --------------------------------------------------------------------------------------
def _key_order(k):
    k = str(k)
    return (0, int(k), "") if k.isdigit() else (1, 0, k)


def _children(d):
    if isinstance(d, list):
        return [(str(i), v) for i, v in enumerate(d) if v is not None]
    return sorted(d.items(), key=lambda kv: _key_order(kv[0]))


def _samples(key, d, depth=0):
    """Yield (key, sample_dict) for every dict under d that carries at least one channel."""
    if isinstance(d, dict) and any(ch in d for ch in CHANNELS):
        yield key, d
    elif isinstance(d, (dict, list)) and depth < 3:
        for k, v in _children(d):
            yield from _samples(k, v, depth + 1)


def extract_samples(rel, data, vkey="vitals"):
    """rel = event path relative to the vitals root, as a list. Returns [(pid, key, sample)]."""
    out = []
    if data is None:
        return out
    if len(rel) == 0:
        if isinstance(data, dict):
            for pid, sub in data.items():
                out += extract_samples([pid], sub, vkey)
    elif len(rel) == 1:
        if isinstance(data, dict) and vkey in data:
            out += [(rel[0], k, s) for k, s in _samples(vkey, data[vkey])]
    elif rel[1] == vkey:
        out += [(rel[0], k, s) for k, s in _samples(rel[-1] if len(rel) > 2 else vkey, data)]
    return out


def command_pids(rel, data):
    """Pids named by a command-node event (handoff / clear). Ignores deletions."""
    if data is None or data is False:
        return []
    if len(rel) >= 1:
        return [rel[0]]
    if isinstance(data, dict):
        return [k for k, v in data.items() if v]
    return []


class SeenSet:
    """Bounded recently-seen-ids set: de-duplicates samples re-delivered after a reconnect."""

    def __init__(self, n=1000):
        self.q, self.s = deque(maxlen=n), set()

    def add(self, x):
        """True if new."""
        if x in self.s:
            return False
        if len(self.q) == self.q.maxlen:
            self.s.discard(self.q[0])
        self.q.append(x)
        self.s.add(x)
        return True


def to_msg(sample):
    """Sample dict -> clean {channel: float}; missing/NaN/non-numeric channels are dropped."""
    msg = {}
    for ch in CHANNELS:
        try:
            v = float(sample.get(ch))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            msg[ch] = v
    if "t" in sample:
        msg["t"] = sample["t"]
    return msg


def spell(pid):
    return " ".join(re.findall(r"[A-Za-z]+|\d", str(pid)))      # "P001" -> "P 0 0 1"


# --------------------------------------------------------------------------------------
# Pipeline: vitals in -> LiveForecastService -> speech + Firebase out
# --------------------------------------------------------------------------------------
class Pipeline:
    def __init__(self, fc, write, say, delete=lambda path: None, horizon=20, step=5, recent=30,
                 fill_missing=True, repeat_s=60, stale_s=15, forecast_root="ambulink/forecast",
                 vkey="vitals", max_batch=50,
                 cmd_roots=None, dashboard_root=None, names=None):
        self.write, self.say, self.delete = write, say, delete   # write(path, payload)
        self.cmd_roots = cmd_roots or {"handoff": "ambulink/handoff", "clear": "ambulink/clear"}
        self.repeat_s, self.stale_s = repeat_s, stale_s
        self.forecast_root, self.vkey, self.max_batch = forecast_root, vkey, max_batch
        self.dashboard_root, self.names = dashboard_root, names
        self.measured = {}      # pid -> channels the sensor actually sent in the last reading
        self.svc = LiveForecastService(fc, self._publish, speak=lambda _t: None,
                                       horizon=horizon, recent=recent, step=step,
                                       fill_missing=fill_missing)
        self.seen = {}          # pid -> SeenSet
        self.last_seen = {}     # pid -> monotonic time of last accepted reading
        self.stale = set()
        self.level = {}         # pid -> None | "warning" | "critical"   (latched alert level)
        self.info = {}          # pid -> last verdict that actually named a channel
        self.last_spoken = {}

    # svc publishes MQTT-style topics; translate them into Firebase paths
    def _publish(self, topic, p):
        parts = topic.split("/")
        if parts[:2] == ["ambulink", "amb"] and parts[-1] == "forecast":
            if not p.get("ready") and p.get("have", 0) % 5:
                return                                         # throttle warm-up writes
            self.write(f"{self.forecast_root}/{parts[2]}", p)
            if self.dashboard_root:
                pid = parts[2]
                self.write(f"{self.dashboard_root}/{pid}",
                           dashboard_view(pid, p, self.measured.get(pid), self.names, CRITICAL_SCORE))
        elif parts[:2] == ["ambulink", "icu"]:
            self.write(topic, p)
        else:
            log.warning("%s: %s", topic, p)

    # ---- vitals ---------------------------------------------------------------------
    def on_vitals_event(self, rel, data):
        found = extract_samples(rel, data, self.vkey)
        if len(found) > self.max_batch:
            log.info("ignoring bulk history (%d samples) at /%s", len(found), "/".join(rel))
            return
        for pid, key, sample in found:
            msg = to_msg(sample)
            if not any(ch in msg for ch in CHANNELS):
                continue
            ident = (key, msg.get("t"), tuple(msg.get(c) for c in CHANNELS))
            if not self.seen.setdefault(pid, SeenSet()).add(ident):
                continue
            self._ingest(pid, msg)

    def _ingest(self, pid, msg):
        if pid in self.svc.handed_off:
            return
        self.last_seen[pid] = time.monotonic()
        if pid in self.stale:
            self.stale.discard(pid)
            self.say(f"Vitals from patient {spell(pid)} restored.")
        self.measured[pid] = [c for c in CHANNELS if c in msg]
        v = self.svc.on_vitals(pid, msg)
        if not v or not v.get("ready"):
            return
        if v.get("channel"):
            self.info[pid] = v                                 # remember the real (unlatched) cause
        if not v["will_be_anomalous"]:
            return
        lvl = "critical" if v.get("future_anomaly_score", 0) >= CRITICAL_SCORE else "warning"
        prev = self.level.get(pid)
        if prev is None or (prev == "warning" and lvl == "critical"):
            self.level[pid] = lvl
            self._announce(pid, self.info.get(pid, v), lvl)

    def _announce(self, pid, v, lvl, reminder=False):
        ch = v.get("channel")
        name, unit, dec = SPOKEN.get(ch, (ch or "vitals", "", 0))
        pc = (v.get("per_channel") or {}).get(ch, {})
        end, med = pc.get("forecast_end"), pc.get("baseline_median")
        direction = "rise" if (end is not None and med is not None and end > med) else "fall"
        ahead = v.get("first_anomalous_step")
        when = (f"in about {ahead} second{'s' if ahead != 1 else ''}" if ahead else "soon")
        name_cap = name[:1].upper() + name[1:]
        if reminder:
            text = f"Reminder. Patient {spell(pid)}. Alert still active. {name_cap} abnormal."
        else:
            head = "Critical alert" if lvl == "critical" else "Warning"
            text = f"{head}. Patient {spell(pid)}. {name_cap} is predicted to {direction} abnormally {when}."
            if end is not None:
                text += f" Forecast {end:.{dec}f} {unit}."
        self.last_spoken[pid] = time.monotonic()
        self.say(text)

    # ---- commands -------------------------------------------------------------------
    def on_handoff(self, pid):
        if pid not in self.svc.pats:
            log.warning("handoff for unknown patient %s", pid)
            return
        if self.svc.on_handoff(pid):
            self.level.pop(pid, None)
            if self.dashboard_root:
                self.write(f"{self.dashboard_root}/{pid}", handed_off_view(pid, self.names))
            self.say(f"Patient {spell(pid)} incoming to ICU.")

    def on_clear(self, pid):
        buf = self.svc.pats.get(pid)
        if buf is None:
            return
        # Clear the latch AND the buffer: the new baseline must come from the post-clear state,
        # otherwise the old crash is still in the window and re-alerts immediately.
        buf.raw.clear()
        buf.n, buf.last_verdict, buf.last_anom = 0, None, False
        self.level.pop(pid, None)
        self.info.pop(pid, None)
        self.say(f"Alert cleared for patient {spell(pid)}. Re-learning baseline.")

    # ---- periodic -------------------------------------------------------------------
    def tick(self):
        now = time.monotonic()
        for pid, seen in list(self.last_seen.items()):
            if pid in self.svc.handed_off:
                continue
            if now - seen > self.stale_s and pid not in self.stale:
                self.stale.add(pid)
                self.say(f"Warning. No vitals received from patient {spell(pid)} "
                         f"for {int(now - seen)} seconds. Check sensor and connection.")
        for pid, lvl in list(self.level.items()):
            if lvl and now - self.last_spoken.get(pid, 0) >= self.repeat_s:
                self._announce(pid, self.info.get(pid, {}), lvl, reminder=True)


# --------------------------------------------------------------------------------------
# Firebase wiring
# --------------------------------------------------------------------------------------
def make_listener(inq, kind, skip_first=True):
    """Return a firebase-admin callback that normalises events onto the work queue."""
    state = {"first": skip_first}

    def cb(event):
        if state["first"]:                       # initial snapshot of existing data: not live
            state["first"] = False
            log.info("%s listener attached", kind)
            return
        path = [p for p in (event.path or "/").split("/") if p]
        if event.event_type == "patch" and isinstance(event.data, dict):
            for k, v in event.data.items():      # patch = several relative puts
                inq.put((kind, path + [p for p in k.split("/") if p], v))
        else:
            inq.put((kind, path, event.data))
    return cb


# --------------------------------------------------------------------------------------
# "Latest values" layout: the ESP32 overwrites one node per ambulance, e.g.
#   ambulances/AMB101 = {temperatureC: 36.9, heartRate: 82, spo2: 97, status: "OK", ...}
# We mirror that tree and sample each ambulance at a fixed rate while it is updating.
# --------------------------------------------------------------------------------------
LATEST_FIELDS = {"temp_c": "temperatureC", "hr_bpm": "heartRate", "spo2": "spo2"}
BAD_STATUS = ("NO FINGER", "NO_FINGER", "ERROR", "INVALID")


def latest_to_sample(node, fields=LATEST_FIELDS):
    """One ambulance node -> {channel: value}; invalid readings are left out (service fills them)."""
    if not isinstance(node, dict):
        return {}
    out = {}
    status = str(node.get("status", "")).upper()
    pulse_ok = not any(b in status for b in BAD_STATUS)
    for ch, key in fields.items():
        v = node.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        if ch == "temp_c" and node.get("temperatureValid") is False:
            continue
        if ch in ("hr_bpm", "spo2") and (not pulse_ok or v <= 0):
            continue
        out[ch] = v
    return out


class LatestAdapter:
    def __init__(self, ingest, period=1.0, hold=10.0, fields=LATEST_FIELDS):
        self.ingest, self.period, self.hold, self.fields = ingest, period, hold, fields
        self.tree, self.last_event, self.last_emit = {}, {}, {}
        self.loaded = False

    def _set(self, path, value):
        if not path:
            self.tree = value if isinstance(value, dict) else {}
            return
        node = self.tree
        for k in path[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        if value is None:
            node.pop(path[-1], None)
        else:
            node[path[-1]] = value

    def on_event(self, rel, data, now=None):
        now = time.monotonic() if now is None else now
        self._set(rel, data)
        if not self.loaded and not rel:          # initial snapshot: load only, don't sample
            self.loaded = True
            return
        self.loaded = True
        pids = [rel[0]] if rel else list(self.tree)
        for pid in pids:
            self.last_event[pid] = now

    def flush(self, now=None):
        now = time.monotonic() if now is None else now
        for pid, t_ev in list(self.last_event.items()):
            if now - t_ev > self.hold:           # ESP32 went quiet: stop (stale warning follows)
                continue
            if now - self.last_emit.get(pid, -1e9) < self.period:
                continue
            self.last_emit[pid] = now
            msg = latest_to_sample(self.tree.get(pid), self.fields)
            if msg:
                msg["t"] = int(time.time())
                self.ingest(pid, msg)


def worker(inq, pipe, stop, latest=None):
    while not stop.is_set():
        try:
            kind, rel, data = inq.get(timeout=0.2 if latest else 1.0)
        except queue.Empty:
            if latest:
                latest.flush()
            pipe.tick()
            continue
        try:
            if kind == "vitals":
                pipe.on_vitals_event(rel, data)
            elif kind == "latest":
                latest.on_event(rel, data)
            else:
                for pid in command_pids(rel, data):
                    (pipe.on_handoff if kind == "handoff" else pipe.on_clear)(pid)
                    pipe.delete(f"{pipe.cmd_roots[kind]}/{pid}")
        except Exception:
            log.exception("error handling %s event %s", kind, rel)
        if latest:
            latest.flush()
        pipe.tick()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lstm_model", help="stem of lstm_model.keras / .scaler.joblib")
    ap.add_argument("--cred", default=os.environ.get("FIREBASE_CRED", "serviceAccountKey.json"))
    ap.add_argument("--db-url", default=os.environ.get("FIREBASE_DB_URL"))
    ap.add_argument("--api-key", default=os.environ.get("FIREBASE_API_KEY"),
                    help="Web API key: connect with it instead of a service account")
    ap.add_argument("--email", default=os.environ.get("FIREBASE_EMAIL"),
                    help="with --api-key: sign in as this user instead of anonymously")
    ap.add_argument("--password", default=os.environ.get("FIREBASE_PASSWORD"))
    ap.add_argument("--vitals-root", default="ambulink/amb", help="parent of {pid}/vitals")
    ap.add_argument("--vitals-key", default="vitals")
    ap.add_argument("--forecast-root", default="ambulink/forecast")
    ap.add_argument("--handoff-root", default="ambulink/handoff")
    ap.add_argument("--clear-root", default="ambulink/clear")
    ap.add_argument("--latest-root", default=os.environ.get("AMBULINK_LATEST_ROOT"),
                    help='read the "latest values" layout from here, e.g. ambulances '
                         "(ambulances/AMB101 = {temperatureC, heartRate, spo2, status})")
    ap.add_argument("--dashboard-root", default="predictions",
                    help="flat dashboard record per patient goes to {this}/{pid}; '' to disable")
    ap.add_argument("--temp-field", default="temperatureC")
    ap.add_argument("--hr-field", default="heartRate")
    ap.add_argument("--spo2-field", default="spo2")
    ap.add_argument("--horizon", type=int, default=20, help="seconds ahead to forecast")
    ap.add_argument("--recent", type=int, default=30, help="samples fed to the forecaster")
    ap.add_argument("--step", type=int, default=5, help="forecast every N readings")
    ap.add_argument("--no-fill", action="store_true", help="require all channels; skip incomplete readings")
    ap.add_argument("--repeat", type=int, default=60, help="seconds between reminders while alert latched")
    ap.add_argument("--stale", type=int, default=15, help="seconds without data before a 'no vitals' warning")
    ap.add_argument("--voice", default="en")
    ap.add_argument("--rate", type=int, default=150)
    ap.add_argument("--aplay-device", default=None, help='ALSA device, e.g. "plughw:2,0" (see aplay -l)')
    ap.add_argument("--mute", action="store_true", help="log alerts, don't speak")
    ap.add_argument("--test-voice", action="store_true", help="speak a test phrase and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)   # its debug lines print the auth token

    speaker = Speaker(a.voice, a.rate, aplay_device=a.aplay_device, enabled=not a.mute)
    if a.test_voice:
        speaker.say("Ambulink voice test. Patient P 0 0 1. Warning. Heart rate predicted to rise abnormally.")
        speaker.wait()
        return
    if not a.db_url:
        ap.error("--db-url (or FIREBASE_DB_URL) is required")

    if a.api_key:
        from firebase_rest import ApiKeyDB, FirebaseAuthError
        try:
            db = ApiKeyDB(a.api_key, a.db_url, a.email, a.password)
            db.reference(a.forecast_root).get()          # fail fast on bad URL / rules
        except (FirebaseAuthError, RuntimeError, OSError) as e:
            log.error("%s", e)
            raise SystemExit(1)
        log.info("connected to %s with API key", a.db_url)
    else:
        if not os.path.exists(a.cred):
            ap.error(f"{a.cred} not found: pass --api-key, or --cred <service account json>")
        import firebase_admin
        from firebase_admin import credentials, db
        firebase_admin.initialize_app(credentials.Certificate(a.cred), {"databaseURL": a.db_url})

    log.info("loading model %s ...", a.model)
    fc = HSPForecaster.load(a.model)
    writer = FirebaseWriter(db)
    pipe = Pipeline(fc, writer.set, speaker.say, delete=writer.delete,
                    horizon=a.horizon, step=a.step, recent=a.recent,
                    fill_missing=not a.no_fill, repeat_s=a.repeat, stale_s=a.stale,
                    forecast_root=a.forecast_root, vkey=a.vitals_key,
                    cmd_roots={"handoff": a.handoff_root, "clear": a.clear_root},
                    dashboard_root=a.dashboard_root or None,
                    names={"temp_c": a.temp_field, "hr_bpm": a.hr_field, "spo2": a.spo2_field})

    inq, stop = queue.Queue(), threading.Event()
    latest = None
    if a.latest_root:
        fields = {"temp_c": a.temp_field, "hr_bpm": a.hr_field, "spo2": a.spo2_field}
        latest = LatestAdapter(pipe._ingest, period=1.0, hold=max(3.0, a.stale - 5), fields=fields)
        log.info("reading latest values from /%s (%s)", a.latest_root, fields)
    threading.Thread(target=worker, args=(inq, pipe, stop, latest), daemon=True, name="worker").start()

    if latest:
        vit = db.reference(a.latest_root).listen(make_listener(inq, "latest", skip_first=False))
    else:
        vit = db.reference(a.vitals_root).listen(make_listener(inq, "vitals"))
    regs = [vit,
            db.reference(a.handoff_root).listen(make_listener(inq, "handoff")),
            db.reference(a.clear_root).listen(make_listener(inq, "clear"))]

    speaker.say("Ambulink monitoring online.")
    log.info("running: need %d samples/patient before the first forecast", pipe.svc.need)

    def _on_signal(*_):
        if stop.is_set():                    # second Ctrl+C: quit immediately
            os._exit(1)
        log.info("stopping ... (Ctrl+C again to force)")
        stop.set()
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    while not stop.is_set():
        stop.wait(0.5)

    # close listeners in the background: an open network stream can block close()
    def _close_all():
        for r in regs:
            try:
                r.close()
            except Exception:
                pass
    t = threading.Thread(target=_close_all, daemon=True)
    t.start()
    t.join(timeout=2)
    writer.stop()
    speaker.stop()
    log.info("stopped")
    logging.shutdown()
    os._exit(0)                              # don't wait on network/TensorFlow threads

if __name__ == "__main__":
    main()
