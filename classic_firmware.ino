/*
 * AMBULINK — Circuit Playground Classic: send onboard temperature to the ESP32 over UART.
 * Library: Adafruit Circuit Playground (Library Manager).
 *
 * Wiring: Classic TX (pin 1) -> ESP32 GPIO16.  GND <-> GND.  Both 3.3V, no level shifter.
 * NOTE: the Classic thermistor reads board/ambient temperature, not body temperature.
 * Fine for a demo; for real patient temp add a DS18B20 probe.
 */
#include <Adafruit_CircuitPlayground.h>

void setup() {
  CircuitPlayground.begin();
  Serial1.begin(9600);   // pin 1 = TX on the Classic
}

void loop() {
  float t = CircuitPlayground.temperature();   // degrees C
  Serial1.println(t, 2);
  delay(1000);
}
