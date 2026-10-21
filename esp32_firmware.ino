/*
 * AMBULINK ESP32 firmware — reads temp from the Circuit Playground Classic over UART,
 * publishes vitals to MQTT at 1 Hz. hr_bpm / spo2 are sent only if you wire real sensors;
 * otherwise the Pi service fills them.
 *
 * Wiring: Classic TX (pin 1) -> ESP32 GPIO16 (RX2), GND <-> GND.
 * Classic sends one CSV line per second, e.g.  "28.40"  (temperature in C)
 *
 * Libraries (Library Manager): WiFi (built in), PubSubClient, ArduinoJson.
 */
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASS";
const char* MQTT_HOST = "192.168.1.50";   // Raspberry Pi IP
const int   MQTT_PORT = 1883;
const char* PATIENT_ID = "P001";

WiFiClient net;
PubSubClient mqtt(net);
String topic;

// Classic feeds temperature on Serial2 (GPIO16 RX)
#define CLASSIC_RX 16
#define CLASSIC_TX 17

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
}

void connectMQTT() {
  while (!mqtt.connected()) {
    if (mqtt.connect(("esp32-" + String(PATIENT_ID)).c_str())) break;
    delay(1000);
  }
}

void setup() {
  Serial.begin(115200);
  Serial2.begin(9600, SERIAL_8N1, CLASSIC_RX, CLASSIC_TX);
  topic = "ambulink/amb/" + String(PATIENT_ID) + "/vitals";
  connectWiFi();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
}

unsigned long lastPub = 0;
float lastTemp = NAN;

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();

  // read the latest temperature line from the Classic (non-blocking)
  while (Serial2.available()) {
    String line = Serial2.readStringUntil('\n');
    line.trim();
    if (line.length()) lastTemp = line.toFloat();
  }

  // publish once per second
  if (millis() - lastPub >= 1000 && !isnan(lastTemp)) {
    lastPub = millis();
    StaticJsonDocument<128> doc;
    doc["t"] = millis() / 1000;
    doc["temp_c"] = lastTemp;
    // doc["hr_bpm"] = <read MAX30102>;   // uncomment when the sensor is wired
    // doc["spo2"]  = <read MAX30102>;
    char buf[128];
    size_t n = serializeJson(doc, buf);
    mqtt.publish(topic.c_str(), buf, n);
  }
}
