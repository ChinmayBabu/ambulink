# AMBULINK LSTM forecaster (+ live ESP32 module)

## Files
- `forecaster.py`: multivariate LSTM (fit / forecast / predict_future_anomaly / save / load)
- `vitals_sim.py`: simulated vitals (stable / deteriorating / critical / glitch)
- `live_service.py`: **live MQTT service** — ESP32 vitals in, forecast + voice + handoff out
- `esp32_firmware.ino`: ESP32 — reads temp over UART, publishes vitals to MQTT
- `classic_firmware.ino`: Circuit Playground Classic — sends temp to the ESP32
- `test_lstm.py`: offline forecaster test
- `test_live.py`: offline live-service test (no broker)

## Train a model first
```python
from forecaster import HSPForecaster, CHANNELS
from vitals_sim import make_stream
cols = lambda a: {ch: a[:,j].tolist() for j,ch in enumerate(CHANNELS)}
fc = HSPForecaster.fit([cols(make_stream("stable", n=300, seed=s)) for s in range(30)], epochs=40)
fc.save("lstm_model")
```
Replace the sim streams with your real healthy captures when you have them.
Edit `CHANNELS` and `FLOOR` at the top of `forecaster.py` to match your streams.

## Live run
1. Pi: `sudo apt install mosquitto mosquitto-clients espeak-ng`
2. `pip install numpy scikit-learn tensorflow paho-mqtt joblib`
3. `python3 live_service.py --model lstm_model --broker 127.0.0.1`
   - `--step N`   run a forecast every N readings (default 5; the LSTM rollout is ~20 ms each)
   - `--horizon N` forecast N seconds ahead (default 20)
   - `--no-fill`  require all channels instead of filling missing ones

### Data path
- Classic temp -> ESP32 (UART) -> Wi-Fi -> MQTT `ambulink/amb/P001/vitals`
  `{"t": 1696900000, "temp_c": 36.8}`  (add `hr_bpm`, `spo2` when wired)
- Service publishes verdicts to `ambulink/amb/P001/forecast`
- Handoff: `mosquitto_pub -t ambulink/handoff/P001 -m go` -> ICU gets `ambulink/icu/P001/handoff`
- Watch: `mosquitto_sub -t "ambulink/#" -v`

### Verdict JSON
`will_be_anomalous`, `future_anomaly_score`, `first_anomalous_step`, `channel`, `reason`,
`forecast_summary`, `forecast_trend`, `vitals`, `latency_ms`, `latched`

## Missing channels
- The live service fills channels the hardware doesn't send with resting values (`RESTING` in
  `live_service.py`), so the demo runs with just the Classic thermistor
- A near-constant filled channel can't false-alarm: `FLOOR` sets a minimum band width per channel

## Behaviour notes
- **Interval forecasting**: a full forecast runs every `--step` readings; other readings echo the
  last verdict. This keeps up with 1 Hz input (the rollout is the costly part)
- **Latched alerts**: once an anomaly is raised it stays raised until a clinician clears it —
  a crashed signal that plateaus won't silently look normal again
- **Early-baseline band**: the "normal" comes from the first third of the recent window, so a
  crash already underway isn't compared against itself

## Known limits (unchanged from the forecaster)
- Trained on healthy data only, so it can't forecast a fault shape it never saw
  - Fast crashes are caught; slow drift is weaker — add real/synthetic degradation to training
- Latency ~20 ms per forecast on a dev box; expect more on a Pi — keep `--step` >= 5, or shorten horizon
- Simulated data only; never run against a real broker, ESP32, or Classic here
- Clinical floors/fills are demo values, not medical guidance
