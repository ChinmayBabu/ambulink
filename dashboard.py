"""Turn a raw forecaster verdict into a flat, frontend-friendly record.

firebase_service.py writes it to   {dashboard_root}/{pid}   (default: predictions/AMB101)

Example record
{
  "patientId": "AMB101",
  "status": "warning",                 # warming_up | stable | warning | critical | handed_off
  "alert": true,
  "latched": false,                    # true = alert held until a clinician clears it
  "riskScore": 84,                     # 0-100
  "alertVital": "heartRate",           # which vital triggered it (null if none)
  "secondsUntilAbnormal": 6,
  "message": "Heart rate predicted to rise abnormally in about 6 s (forecast 118 bpm)",
  "recommendation": "monitor",         # stable | monitor | prepare_crash_cart
  "recommendationText": "Alert ER team, prepare monitored bed",
  "current":   {"temperatureC": 36.9, "heartRate": 96,  "spo2": null},   # null = not measured
  "predicted": {"temperatureC": 37.0, "heartRate": 118, "spo2": null},
  "trend":     {"temperatureC": "steady", "heartRate": "rising", "spo2": null},
  "horizonSeconds": 20,
  "warmup": null,                      # {"have": 12, "need": 30} while warming up
  "latencyMs": 21.4,
  "updatedAt": 1792053529000           # epoch ms
}
Recommendations are demo values, not medical guidance.
"""
import time

DEFAULT_NAMES = {"temp_c": "temperatureC", "hr_bpm": "heartRate", "spo2": "spo2"}
LABEL = {"temp_c": ("Temperature", "°C", 1), "hr_bpm": ("Heart rate", "bpm", 0),
         "spo2": ("SpO2", "%", 0)}
TREND_EPS = {"temp_c": 0.1, "hr_bpm": 2.0, "spo2": 0.5}
RECOMMEND = {
    "stable": ("stable", "Stable - routine handover"),
    "warning": ("monitor", "Alert ER team, prepare monitored bed"),
    "critical": ("prepare_crash_cart", "Prepare crash cart and resus bay"),
}


def _r(x, d):
    return None if x is None else round(float(x), d)


def dashboard_view(pid, v, measured=None, names=None, critical_score=0.90):
    """v = verdict from LiveForecastService; measured = channels actually sent by the sensor
    (others were filled with resting values and are shown as null)."""
    names = names or DEFAULT_NAMES
    measured = set(measured) if measured is not None else set(names)
    now_ms = int(time.time() * 1000)
    blank = {names[c]: None for c in names}
    rec = {"patientId": pid, "updatedAt": now_ms, "current": dict(blank),
           "predicted": dict(blank), "trend": dict(blank), "alert": False, "latched": False,
           "riskScore": 0, "alertVital": None, "secondsUntilAbnormal": None,
           "horizonSeconds": None, "warmup": None, "latencyMs": None}

    if not v.get("ready"):
        rec.update(status="warming_up", warmup={"have": v.get("have"), "need": v.get("need")},
                   message=f"Collecting baseline ({v.get('have')}/{v.get('need')} readings)",
                   recommendation="stable", recommendationText="Collecting baseline")
        return rec

    vitals, per = v.get("vitals") or {}, v.get("per_channel") or {}
    trend = v.get("forecast_trend") or {}
    for c, name in names.items():
        if c not in measured:
            continue
        d = LABEL[c][2] if c in LABEL else 2
        rec["current"][name] = _r(vitals.get(c), d)
        rec["predicted"][name] = _r((per.get(c) or {}).get("forecast_end"), d)
        t = trend.get(c)
        if t is not None:
            eps = TREND_EPS.get(c, 0.0)
            rec["trend"][name] = "rising" if t > eps else "falling" if t < -eps else "steady"

    score = float(v.get("future_anomaly_score") or 0)
    alert = bool(v.get("will_be_anomalous"))
    status = ("critical" if score >= critical_score else "warning") if alert else "stable"
    ch = v.get("channel")
    ahead = v.get("first_anomalous_step")
    if alert and ch:
        lab, unit, dec = LABEL.get(ch, (ch, "", 1))
        pc = per.get(ch) or {}
        end, med = pc.get("forecast_end"), pc.get("baseline_median")
        direction = "rise" if (end is not None and med is not None and end > med) else "fall"
        when = f"in about {ahead} s" if ahead else "soon"
        msg = f"{lab} predicted to {direction} abnormally {when}"
        if end is not None:
            msg += f" (forecast {end:.{dec}f} {unit})"
    elif alert:
        msg = "Alert active - awaiting clinician review"
    else:
        msg = "Vitals forecast within normal range"

    code, text = RECOMMEND[status]
    rec.update(status=status, alert=alert, latched=bool(v.get("latched")),
               riskScore=int(round(score * 100)), alertVital=names.get(ch) if ch else None,
               secondsUntilAbnormal=ahead if alert else None, message=msg,
               recommendation=code, recommendationText=text,
               horizonSeconds=v.get("horizon_steps"), latencyMs=v.get("latency_ms"))
    return rec


def handed_off_view(pid, names=None):
    names = names or DEFAULT_NAMES
    blank = {names[c]: None for c in names}
    return {"patientId": pid, "status": "handed_off", "alert": False, "latched": False,
            "riskScore": 0, "alertVital": None, "secondsUntilAbnormal": None,
            "message": "Handed off to ICU", "recommendation": "stable",
            "recommendationText": "Handed off to ICU", "current": blank, "predicted": dict(blank),
            "trend": dict(blank), "horizonSeconds": None, "warmup": None, "latencyMs": None,
            "updatedAt": int(time.time() * 1000)}