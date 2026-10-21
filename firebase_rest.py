"""Firebase Realtime Database over REST, authenticated with the Web API key.

Drop-in stand-in for `firebase_admin.db` as used by firebase_service.py:
    db = ApiKeyDB(api_key, db_url)            # or ApiKeyDB(api_key, db_url, email, password)
    db.reference("a/b").set({...})
    db.reference("a/b").delete()
    reg = db.reference("a").listen(callback)  # callback(event) with .event_type .path .data
    reg.close()

Sign-in: anonymous (enable Authentication -> Sign-in method -> Anonymous), or email/password
if given. The ID token lasts 1 hour; it is refreshed automatically.
Rules must allow the signed-in user, e.g.  {"rules": {"ambulink": {".read": "auth != null",
".write": "auth != null"}}}
"""
import json
import logging
import threading
import time
from types import SimpleNamespace

import requests

log = logging.getLogger("ambulink.firebase")

SIGNUP = "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={}"
SIGNIN = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={}"
REFRESH = "https://securetoken.googleapis.com/v1/token?key={}"


class FirebaseAuthError(RuntimeError):
    pass


def _err(r):
    try:
        return r.json().get("error", {}).get("message") or r.text
    except ValueError:
        return r.text


class ApiKeyAuth:
    """Holds an ID token for the API key and refreshes it before it expires."""

    def __init__(self, api_key, email=None, password=None, timeout=15):
        self.key, self.email, self.password, self.timeout = api_key, email, password, timeout
        self._lock = threading.Lock()
        self._id = self._refresh_tok = None
        self._exp = 0.0
        self._sign_in()

    def _store(self, id_token, refresh_token, expires_in):
        self._id, self._refresh_tok = id_token, refresh_token
        self._exp = time.time() + int(expires_in) - 300          # renew 5 min early

    def _sign_in(self):
        if self.email:
            url, body = SIGNIN.format(self.key), {"email": self.email, "password": self.password,
                                                  "returnSecureToken": True}
        else:
            url, body = SIGNUP.format(self.key), {"returnSecureToken": True}
        r = requests.post(url, json=body, timeout=self.timeout)
        if r.status_code != 200:
            msg = _err(r)
            hint = (" -> enable Anonymous in Authentication > Sign-in method"
                    if "ADMIN_ONLY_OPERATION" in msg or "OPERATION_NOT_ALLOWED" in msg else
                    " -> check the API key (Project settings > General > Web API Key)"
                    if "API key" in msg or "API_KEY" in msg else "")
            raise FirebaseAuthError(f"Firebase sign-in failed: {msg}{hint}")
        d = r.json()
        self._store(d["idToken"], d["refreshToken"], d.get("expiresIn", 3600))
        log.info("signed in to Firebase (%s)", "email" if self.email else "anonymous")

    def _refresh(self):
        r = requests.post(REFRESH.format(self.key), timeout=self.timeout,
                          data={"grant_type": "refresh_token", "refresh_token": self._refresh_tok})
        if r.status_code != 200:
            log.warning("token refresh failed (%s); signing in again", _err(r))
            self._sign_in()
            return
        d = r.json()
        self._store(d["id_token"], d["refresh_token"], d.get("expires_in", 3600))
        log.info("Firebase token refreshed")

    def token(self, force=False):
        with self._lock:
            if force or time.time() >= self._exp:
                self._refresh()
            return self._id


class _Registration:
    def __init__(self):
        self._stop = threading.Event()
        self._resp = None
        self.thread = None

    def close(self):
        self._stop.set()
        r = self._resp
        if r is not None:
            try:
                r.close()
            except Exception:
                pass


class Reference:
    def __init__(self, db, path):
        self.db, self.path = db, path.strip("/")

    def _url(self):
        return f"{self.db.url}/{self.path}.json"

    def _call(self, method, **kw):
        refreshed, tries = False, 0
        while True:
            try:
                r = self.db.session().request(method, self._url(), timeout=self.db.timeout,
                                              params={"auth": self.db.auth.token(force=refreshed)},
                                              **kw)
            except requests.RequestException as e:      # Wi-Fi blip / TLS timeout: retry
                tries += 1
                if tries > self.db.retries:
                    raise
                log.warning("%s /%s: %s; retry %d/%d", method, self.path,
                            type(e).__name__, tries, self.db.retries)
                self.db.reset_session()
                time.sleep(min(2 ** tries, 10))
                continue
            if r.status_code == 401 and not refreshed:
                refreshed = True                        # token expired -> refresh once
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{method} /{self.path}: {r.status_code} {_err(r)}")
            return r.json() if r.content else None

    def get(self):
        return self._call("GET")

    def set(self, value):
        self._call("PUT", data=json.dumps(value))

    def push(self, value):
        return self._call("POST", data=json.dumps(value))

    def delete(self):
        self._call("DELETE")

    def listen(self, callback):
        """Streams changes (server-sent events) on a background thread. Reconnects on errors and
        token expiry. The first event delivered is the initial snapshot (like firebase_admin);
        snapshots re-sent after a reconnect are skipped."""
        reg = _Registration()
        reg.thread = threading.Thread(target=self._stream, args=(callback, reg), daemon=True,
                                      name=f"fb-listen:{self.path}")
        reg.thread.start()
        return reg

    def _stream(self, callback, reg):
        first_connect, backoff = True, 1
        while not reg._stop.is_set():
            skip_snapshot = not first_connect
            try:
                resp = requests.get(self._url(), stream=True, timeout=(self.db.timeout, 90),
                                    params={"auth": self.db.auth.token()},
                                    headers={"Accept": "text/event-stream"})
                reg._resp = resp
                if resp.status_code == 401:
                    self.db.auth.token(force=True)
                    resp.close()
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"listen /{self.path}: {resp.status_code} {_err(resp)}")
                backoff, revoked = 1, False
                for ev in parse_sse(resp.iter_lines(decode_unicode=True)):
                    if reg._stop.is_set():
                        break
                    etype, data = ev
                    if etype in ("put", "patch"):
                        if skip_snapshot and etype == "put":
                            skip_snapshot = False
                            if data.get("path") == "/":
                                continue
                        skip_snapshot = False
                        first_connect = False
                        callback(SimpleNamespace(event_type=etype, path=data.get("path", "/"),
                                                 data=data.get("data")))
                    elif etype == "auth_revoked":
                        log.info("listen /%s: token expired, reconnecting", self.path)
                        self.db.auth.token(force=True)
                        revoked = True
                        break
                    elif etype == "cancel":
                        raise RuntimeError(f"listen /{self.path} cancelled (rules deny read?): {data}")
                resp.close()
                if not revoked:
                    reg._stop.wait(1)                       # stream ended: brief pause, reconnect
            except Exception as e:
                if reg._stop.is_set():
                    break
                log.error("listen /%s: %r; retry in %ds", self.path, e, backoff)
                reg._stop.wait(backoff)
                backoff = min(backoff * 2, 30)


def parse_sse(lines):
    """Yield (event, parsed_data) from an iterable of SSE text lines."""
    etype, buf = None, []
    for line in lines:
        if line is None:
            continue
        if line == "":
            if etype is not None:
                raw = "\n".join(buf)
                try:
                    data = json.loads(raw) if raw else None
                except ValueError:
                    data = raw
                yield etype, data
            etype, buf = None, []
        elif line.startswith("event:"):
            etype = line[6:].strip()
        elif line.startswith("data:"):
            buf.append(line[5:].strip())


class ApiKeyDB:
    def __init__(self, api_key, db_url, email=None, password=None, timeout=15, retries=5):
        self.url = db_url.rstrip("/")
        self.timeout, self.retries = timeout, retries
        self._local = threading.local()                 # one kept-alive connection per thread
        self.auth = ApiKeyAuth(api_key, email, password, timeout)

    def session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = requests.Session()
        return s

    def reset_session(self):
        s = getattr(self._local, "s", None)
        self._local.s = None
        if s is not None:
            try:
                s.close()
            except Exception:
                pass

    def reference(self, path="/"):
        return Reference(self, path)
