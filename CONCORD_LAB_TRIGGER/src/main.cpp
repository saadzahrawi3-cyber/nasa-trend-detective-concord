#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ============================================================================
// 1. WIFI & NETWORK CONFIGURATION
// ============================================================================
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

const uint16_t UDP_LISTEN_PORT = 4210; // Receives telemetry from Streamlit APP.PY
const uint16_t UDP_NOTIFY_PORT = 4211; // Sends instant NFC scan events
WiFiUDP udp;
WebServer server(80);                  // HTTP endpoint for Streamlit auto-sync

// ============================================================================
// 2. PIN DEFINITIONS (THE GARDIANT HARDWARE WIRING)
// ============================================================================
// 0.96" I2C OLED Display
#define OLED_SDA        21
#define OLED_SCL        22
#define SCREEN_WIDTH    128
#define SCREEN_HEIGHT   64
#define OLED_RESET      -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// PN532 NFC Reader (UART2 / HSU Mode: TXD->16, RXD->17)
#define PN532_RX2       16
#define PN532_TX2       17

// Dual IR Sensors (Active LOW on standard LM393 obstacle modules)
#define PIN_IR_OUTER    14  // Door Entry Trigger
#define PIN_IR_INNER    13  // Direction Check
#define IR_ACTIVE_STATE LOW // Change to HIGH if your IR sensors output HIGH on detection

// 5 Tactical LEDs
#define LED_WHITE       4   // bitmask[3]: Inverse Paradox Coupling / Standby
#define LED_BLUE        25  // bitmask[0]: Hydrological Trend Significant (p < 0.05)
#define LED_YELLOW      26  // bitmask[1]: CUSUM Regime Shift (Orange/Yellow slot)
#define LED_GREEN       27  // bitmask[4]: Baseline Stable / NFC Read Flash
#define LED_RED         33  // bitmask[2]: Critical Anomaly (Mann-Kendall p < 0.01)

// ============================================================================
// 3. LAB PRESENCE & DIRECTIONAL IR STATE MACHINE
// ============================================================================
int occupantCount = 0;
bool labOccupied = false;

enum IrStepState { IR_IDLE, IR_OUTER_FIRST, IR_INNER_FIRST };
IrStepState irState = IR_IDLE;
unsigned long irStateTimestamp = 0;
unsigned long lastIrEventTime = 0;
const unsigned long IR_SEQUENCE_TIMEOUT_MS = 1500;
const unsigned long IR_COOLDOWN_MS = 800;

// ============================================================================
// 4. NFC REGIONAL PROFILES & STATE
// ============================================================================
struct NfcRegionProfile {
  const char* knownUid;    // Optional fixed UID; unknown cards cycle profiles automatically
  const char* regionKey;   // Key read by Streamlit APP.PY
  const char* regionTitle; // Short title for OLED
  const char* windowMode;  // "FULL_35Y" or "SHORT_12Y"
};

const NfcRegionProfile REGION_PROFILES[] = {
  {"04:A1:B2:C3", "BERRECHID",  "Berrechid Plain",  "FULL_35Y"},
  {"04:D4:E5:F6", "CASABLANCA", "Casablanca Urban", "FULL_35Y"},
  {"04:11:22:33", "SEBOU",      "Sebou Estuary",    "FULL_35Y"},
  {"04:44:55:66", "OUARZAZATE", "Ouarzazate Arid",  "FULL_35Y"},
  {"04:77:88:99", "AMAZON",     "Amazon Basin",     "FULL_35Y"},
  {"04:AA:BB:CC", "SVALBARD",   "Svalbard Arctic",  "SHORT_12Y"}
};
const int NUM_PROFILES = sizeof(REGION_PROFILES) / sizeof(REGION_PROFILES[0]);

String lastNfcUid = "NONE";
String activeRegionKey = "BERRECHID";
String activeRegionTitle = "Berrechid Plain";
String activeWindowMode = "FULL_35Y";
uint32_t nfcScanCounter = 0;
int nextProfileIndex = 0;
unsigned long lastNfcScanMs = 0;
bool pn532Ready = false;

// ============================================================================
// 5. SCIENTIFIC TELEMETRY STATE (FROM STREAMLIT APP.PY)
// ============================================================================
String targetVar  = "GWETTOP";
String slopeDec   = "-0.004";
String driverVar  = "T2M";
String corrVal    = "-0.68";
String tipYear    = "2016";
String predEnd    = "0.210";
String ledBitmask = "00000";
bool hasTelemetry = false;

unsigned long lastOledPageSwitch = 0;
int oledPage = 0;

// ============================================================================
// 6. HELPER FUNCTIONS: LED CONTROL
// ============================================================================
void setAllLedsLow() {
  digitalWrite(LED_BLUE,   LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED,    LOW);
  digitalWrite(LED_WHITE,  LOW);
  digitalWrite(LED_GREEN,  LOW);
}

void applyTelemetryBitmask() {
  if (!labOccupied) {
    setAllLedsLow();
    return;
  }
  if (ledBitmask.length() >= 5) {
    digitalWrite(LED_BLUE,   ledBitmask[0] == '1' ? HIGH : LOW);
    digitalWrite(LED_YELLOW, ledBitmask[1] == '1' ? HIGH : LOW);
    digitalWrite(LED_RED,    ledBitmask[2] == '1' ? HIGH : LOW);
    digitalWrite(LED_WHITE,  ledBitmask[3] == '1' ? HIGH : LOW);
    digitalWrite(LED_GREEN,  ledBitmask[4] == '1' ? HIGH : LOW);
  }
}

void flashAuthorizedLed() {
  setAllLedsLow();
  for (int i = 0; i < 2; i++) {
    digitalWrite(LED_GREEN, HIGH);
    delay(120);
    digitalWrite(LED_GREEN, LOW);
    delay(80);
  }
  applyTelemetryBitmask();
}

// ============================================================================
// 7. NATIVE PN532 UART2 (HSU) DRIVER (NO LIBRARY CONFLICTS)
// ============================================================================
bool pn532WakeAndConfig() {
  // Send HSU Wakeup preamble + SAMConfiguration (Normal Mode, No IRQ)
  const uint8_t wakeSamCmd[] = {
    0x55, 0x55, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0xFF, 0x03, 0xFD, 0xD4, 0x14, 0x01, 0x17, 0x00
  };
  while (Serial2.available()) Serial2.read();
  Serial2.write(wakeSamCmd, sizeof(wakeSamCmd));
  Serial2.flush();

  unsigned long start = millis();
  uint8_t resp[32];
  int idx = 0;
  while (millis() - start < 400 && idx < (int)sizeof(resp)) {
    if (Serial2.available()) {
      resp[idx++] = Serial2.read();
    }
  }
  // Look for SAMConfiguration response frame code 0xD5 0x15
  for (int i = 0; i < idx - 1; i++) {
    if (resp[i] == 0xD5 && resp[i + 1] == 0x15) {
      return true;
    }
  }
  return false;
}

bool pn532ReadPassiveTargetUID(String &uidOut) {
  // InListPassiveTarget (MaxTg=1, BrTy=0x00 for 106 kbps ISO14443A)
  const uint8_t listCmd[] = {
    0x00, 0x00, 0xFF, 0x04, 0xFC, 0xD4, 0x4A, 0x01, 0x00, 0xE1, 0x00
  };
  while (Serial2.available()) Serial2.read();
  Serial2.write(listCmd, sizeof(listCmd));

  unsigned long start = millis();
  uint8_t buf[48];
  int len = 0;
  while (millis() - start < 90 && len < (int)sizeof(buf)) {
    if (Serial2.available()) {
      buf[len++] = Serial2.read();
    }
  }

  // Locate response code 0xD5 0x4B (InListPassiveTarget response)
  for (int i = 0; i < len - 7; i++) {
    if (buf[i] == 0xD5 && buf[i + 1] == 0x4B && buf[i + 2] >= 1) {
      uint8_t uidLen = buf[i + 7];
      if (uidLen >= 4 && uidLen <= 10 && (i + 8 + uidLen) <= len) {
        String formatted = "";
        for (uint8_t k = 0; k < uidLen; k++) {
          if (k > 0) formatted += ":";
          if (buf[i + 8 + k] < 0x10) formatted += "0";
          formatted += String(buf[i + 8 + k], HEX);
        }
        formatted.toUpperCase();
        uidOut = formatted;
        return true;
      }
    }
  }
  return false;
}

// ============================================================================
// 8. DUAL IR SENSOR PRESENCE GATE LOGIC
// ============================================================================
void updateIrPresenceGate() {
  unsigned long now = millis();
  if (now - lastIrEventTime < IR_COOLDOWN_MS) return;

  bool outerTrig = (digitalRead(PIN_IR_OUTER) == IR_ACTIVE_STATE);
  bool innerTrig = (digitalRead(PIN_IR_INNER) == IR_ACTIVE_STATE);

  if (irState == IR_IDLE) {
    if (outerTrig && !innerTrig) {
      irState = IR_OUTER_FIRST;
      irStateTimestamp = now;
    } else if (innerTrig && !outerTrig) {
      irState = IR_INNER_FIRST;
      irStateTimestamp = now;
    }
  } else if (irState == IR_OUTER_FIRST) {
    if (innerTrig) {
      // Sequence: Outer (14) -> Inner (13) => PERSON ENTERED LAB
      occupantCount++;
      labOccupied = true;
      irState = IR_IDLE;
      lastIrEventTime = now;
      Serial.printf("[IR GATE] Person ENTERED lab. Occupants: %d\n", occupantCount);
      applyTelemetryBitmask();
    } else if (now - irStateTimestamp > IR_SEQUENCE_TIMEOUT_MS) {
      irState = IR_IDLE;
    }
  } else if (irState == IR_INNER_FIRST) {
    if (outerTrig) {
      // Sequence: Inner (13) -> Outer (14) => PERSON EXITED LAB
      if (occupantCount > 0) occupantCount--;
      labOccupied = (occupantCount > 0);
      irState = IR_IDLE;
      lastIrEventTime = now;
      Serial.printf("[IR GATE] Person EXITED lab. Occupants: %d\n", occupantCount);
      applyTelemetryBitmask();
    } else if (now - irStateTimestamp > IR_SEQUENCE_TIMEOUT_MS) {
      irState = IR_IDLE;
    }
  }
}

// ============================================================================
// 9. NFC SCAN HANDLER & REGION MAPPER
// ============================================================================
void pollNfcScanner() {
  if (!pn532Ready) return;
  if (millis() - lastNfcScanMs < 1500) return;

  String scannedUid = "";
  if (pn532ReadPassiveTargetUID(scannedUid)) {
    lastNfcScanMs = millis();
    lastNfcUid = scannedUid;
    nfcScanCounter++;

    // Ensure lab wakes up if someone badges in directly
    if (!labOccupied) {
      occupantCount = 1;
      labOccupied = true;
    }

    // Match known UID or cycle through the 6 NASA regional profiles
    bool matched = false;
    for (int i = 0; i < NUM_PROFILES; i++) {
      if (scannedUid.equalsIgnoreCase(REGION_PROFILES[i].knownUid)) {
        activeRegionKey   = REGION_PROFILES[i].regionKey;
        activeRegionTitle = REGION_PROFILES[i].regionTitle;
        activeWindowMode  = REGION_PROFILES[i].windowMode;
        matched = true;
        break;
      }
    }
    if (!matched) {
      activeRegionKey   = REGION_PROFILES[nextProfileIndex].regionKey;
      activeRegionTitle = REGION_PROFILES[nextProfileIndex].regionTitle;
      activeWindowMode  = REGION_PROFILES[nextProfileIndex].windowMode;
      nextProfileIndex  = (nextProfileIndex + 1) % NUM_PROFILES;
    }

    Serial.printf("[NFC SCAN] UID=%s -> Region=%s (%s)\n",
                  lastNfcUid.c_str(), activeRegionKey.c_str(), activeWindowMode.c_str());

    flashAuthorizedLed();

    // Broadcast UDP notification packet on port 4211 for local listeners
    IPAddress broadcastIp = WiFi.localIP();
    broadcastIp[3] = 255;
    String notifyPacket = "NFC|" + lastNfcUid + "|" + activeRegionKey + "|" + activeWindowMode + "|" + String(nfcScanCounter);
    udp.beginPacket(broadcastIp, UDP_NOTIFY_PORT);
    udp.print(notifyPacket);
    udp.endPacket();
  }
}

// ============================================================================
// 10. UDP TELEMETRY PARSER (FROM STREAMLIT APP.PY)
// ============================================================================
void pollUdpTelemetry() {
  int packetSize = udp.parsePacket();
  if (packetSize <= 0) return;

  char buf[256];
  int len = udp.read(buf, sizeof(buf) - 1);
  if (len <= 0) return;
  buf[len] = '\0';
  String payload = String(buf);
  payload.trim();

  // Expected format from APP.PY:
  // TARGET|SLOPE_DEC|STRONGEST_DRIVER|CORR|TIP_YEAR|PRED_END|BITMASK_5LEDS
  int p1 = payload.indexOf('|');
  int p2 = payload.indexOf('|', p1 + 1);
  int p3 = payload.indexOf('|', p2 + 1);
  int p4 = payload.indexOf('|', p3 + 1);
  int p5 = payload.indexOf('|', p4 + 1);
  int p6 = payload.indexOf('|', p5 + 1);

  if (p1 > 0 && p6 > 0) {
    targetVar  = payload.substring(0, p1);
    slopeDec   = payload.substring(p1 + 1, p2);
    driverVar  = payload.substring(p2 + 1, p3);
    corrVal    = payload.substring(p3 + 1, p4);
    tipYear    = payload.substring(p4 + 1, p5);
    predEnd    = payload.substring(p5 + 1, p6);
    ledBitmask = payload.substring(p6 + 1);
    hasTelemetry = true;

    Serial.printf("[UDP TELEMETRY] %s | Bitmask: %s\n", payload.c_str(), ledBitmask.c_str());
    applyTelemetryBitmask();
  }
}

// ============================================================================
// 11. OLED DISPLAY RENDERER (STANDBY VS ACTIVE MULTI-PAGE)
// ============================================================================
void renderOledScreen() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  if (!labOccupied) {
    // Module stays inactive when nobody is inside the lab
    display.setCursor(0, 18);
    display.println("CONCORD LAB STATION");
    display.drawLine(0, 28, 127, 28, SSD1306_WHITE);
    display.setCursor(0, 34);
    display.println("STATUS: STANDBY");
    display.setCursor(0, 46);
    display.println("Awaiting Lab Entry...");
    display.display();
    return;
  }

  // Rotate between 2 telemetry views every 3.5 seconds when Lab is Occupied
  if (millis() - lastOledPageSwitch > 3500) {
    oledPage = (oledPage + 1) % 2;
    lastOledPageSwitch = millis();
  }

  if (oledPage == 0) {
    display.setCursor(0, 0);
    display.print("CONCORD | OCC:");
    display.println(occupantCount);
    display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

    display.setCursor(0, 14);
    display.print(targetVar);
    display.print(": ");
    display.print(slopeDec);
    display.println("/dec");

    display.setCursor(0, 26);
    display.print("DRV: ");
    display.print(driverVar);
    display.print(" r=");
    display.println(corrVal);

    display.setCursor(0, 38);
    display.print("TIP: ");
    display.print(tipYear);
    display.print(" P:");
    display.println(predEnd);

    display.setCursor(0, 52);
    display.print("BITS:[");
    display.print(ledBitmask);
    display.print("] ");
    display.println(activeRegionKey.substring(0, 6));
  } else {
    display.setCursor(0, 0);
    display.println("NFC REGIONAL ROUTER");
    display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

    display.setCursor(0, 15);
    display.print("UID: ");
    display.println(lastNfcUid);

    display.setCursor(0, 28);
    display.print("REG: ");
    display.println(activeRegionTitle);

    display.setCursor(0, 41);
    display.print("WIN: ");
    display.println(activeWindowMode);

    display.setCursor(0, 53);
    display.print("IP: ");
    display.println(WiFi.localIP());
  }

  display.display();
}

// ============================================================================
// 12. HTTP STATUS ENDPOINT FOR STREAMLIT AUTO-SYNC
// ============================================================================
void setupHttpRoutes() {
  server.on("/status", HTTP_GET, []() {
    String json = "{";
    json += "\"lab_occupied\":" + String(labOccupied ? "true" : "false") + ",";
    json += "\"occupants\":" + String(occupantCount) + ",";
    json += "\"nfc_uid\":\"" + lastNfcUid + "\",";
    json += "\"region_key\":\"" + activeRegionKey + "\",";
    json += "\"window_mode\":\"" + activeWindowMode + "\",";
    json += "\"scan_count\":" + String(nfcScanCounter) + ",";
    json += "\"led_bitmask\":\"" + ledBitmask + "\"";
    json += "}";
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", json);
  });
  server.begin();
}

// ============================================================================
// 13. ARDUINO SETUP & MAIN LOOP
// ============================================================================
void setup() {
  Serial.begin(115200);

  // Configure LEDs
  pinMode(LED_WHITE,  OUTPUT);
  pinMode(LED_BLUE,   OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_GREEN,  OUTPUT);
  pinMode(LED_RED,    OUTPUT);
  setAllLedsLow();

  // Configure Dual IR Sensors
  pinMode(PIN_IR_OUTER, INPUT_PULLUP);
  pinMode(PIN_IR_INNER, INPUT_PULLUP);

  // Initialize I2C OLED (SDA=21, SCL=22)
  Wire.begin(OLED_SDA, OLED_SCL);
  if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 10);
    display.println("CONCORD TELEMETRY");
    display.println("Connecting WiFi...");
    display.display();
  }

  // Initialize PN532 UART2 (RX2=16, TX2=17)
  Serial2.begin(115200, SERIAL_8N1, PN532_RX2, PN532_TX2);
  delay(100);
  pn532Ready = pn532WakeAndConfig();
  Serial.printf("[PN532] UART2 Status: %s\n", pn532Ready ? "ONLINE" : "NOT DETECTED");

  // Connect Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 12000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[WIFI] Connected! IP Address: ");
    Serial.println(WiFi.localIP());
    udp.begin(UDP_LISTEN_PORT);
    setupHttpRoutes();
  }
}

void loop() {
  server.handleClient();
  updateIrPresenceGate();
  pollNfcScanner();
  pollUdpTelemetry();
  renderOledScreen();
}