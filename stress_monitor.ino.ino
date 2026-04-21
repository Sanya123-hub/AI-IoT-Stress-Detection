// ============================================================
//  AI-IoT Stress Detection System — ESP32 DevKit V1
//  Sensors : MAX30102 | DHT22 | GSR | MPU-6050
//  Output  : Serial Monitor (CSV format)
//  Interval: 2 seconds per reading
//  Session : 15 minutes = 450 readings per subject
// ============================================================

#include <MAX30105.h>
#include <heartRate.h>
#include <DHT.h>
#include <MPU6050.h>
#include <Wire.h>
#include <math.h>

// ── CHANGE THIS FOR EVERY SUBJECT ───────────────────────────
#define CURRENT_SUBJECT     2        // Change to 1–25 before uploading
#define SUBJECT_TYPE        "Normal" // "Normal" or "Exercise" (subjects 3,7,12,18,22)

// ── SESSION SETTINGS ─────────────────────────────────────────
#define SESSION_DURATION_MIN  15     // Total session time in minutes
#define READING_INTERVAL_MS   2000   // 2 seconds between readings
#define MAX_READINGS          450    // 15 min × 30 readings/min = 450

// ── Pin Definitions ──────────────────────────────────────────
#define DHT_PIN       4       // GPIO4  — DHT22 data
#define GSR_PIN       34      // GPIO34 — GSR analog (ADC1)
#define SDA_PIN       21      // GPIO21 — I2C SDA (MAX30102 + MPU6050)
#define SCL_PIN       22      // GPIO22 — I2C SCL (MAX30102 + MPU6050)
#define ONBOARD_LED   2       // GPIO2  — onboard LED (status indicator)

// ── Stress Thresholds ────────────────────────────────────────
#define HR_NORMAL_MAX     80    // BPM
#define HR_STRESS_MAX     100   // BPM
#define GSR_NORMAL_MIN    1400  // ADC value (higher = less stress)
#define GSR_STRESS_MIN    1100  // ADC value
#define TEMP_STRESS_MIN   29.8  // °C
#define TEMP_HIGH_MIN     30.4  // °C

// ── Sensor Objects ───────────────────────────────────────────
MAX30105  particleSensor;
DHT       dht(DHT_PIN, DHT22);
MPU6050   mpu;

// ── Global Variables ─────────────────────────────────────────
int   readingCount    = 0;
long  lastReadingTime = 0;
long  sessionStart    = 0;
bool  sessionActive   = false;

// Heart rate calculation
const byte  RATE_SIZE = 6;
byte  rates[RATE_SIZE];
byte  rateSpot        = 0;
long  lastBeat        = 0;
float beatsPerMinute  = 0;
int   beatAvg         = 0;

// ── Setup ────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(ONBOARD_LED, OUTPUT);
  digitalWrite(ONBOARD_LED, LOW);

  printHeader();

  // I2C
  Wire.begin(SDA_PIN, SCL_PIN);

  // MAX30102
  initMAX30102();

  // DHT22
  dht.begin();
  Serial.println("  [✓] DHT22 initialized on GPIO4");

  // MPU6050
  initMPU6050();

  // Session Info
  Serial.println();
  Serial.println("════════════════════════════════════════════════════════");
  Serial.print  ("  Subject ID   : "); Serial.println(CURRENT_SUBJECT);
  Serial.print  ("  Subject Type : "); Serial.println(SUBJECT_TYPE);
  Serial.print  ("  Session Time : "); Serial.print(SESSION_DURATION_MIN);
  Serial.println(" minutes");
  Serial.print  ("  Interval     : "); Serial.print(READING_INTERVAL_MS / 1000);
  Serial.println(" seconds");
  Serial.print  ("  Total Reads  : "); Serial.println(MAX_READINGS);
  Serial.println("════════════════════════════════════════════════════════");
  Serial.println();
  Serial.println("  ▶ Place finger firmly on MAX30102 sensor");
  Serial.println("  ▶ Attach GSR electrodes to index + middle finger");
  Serial.println("  ▶ Session starts automatically in 5 seconds...");
  Serial.println();

  // Countdown
  for (int i = 5; i > 0; i--) {
    Serial.print("  Starting in "); Serial.print(i); Serial.println("...");
    blinkLED(1, 300);
    delay(700);
  }

  // CSV Header
  Serial.println();
  Serial.println("── CSV DATA (copy lines starting with CSV>>) ──────────");
  Serial.println("CSV>>Subject_ID,Reading_No,HeartRate_BPM,Temp_C,Humid_Pct,Conductance_uS,Stress_Label,Pitch_Deg,Roll_Deg");
  Serial.println();

  // Table Header
  Serial.println("Reading | HR(BPM) | Temp(°C) | Humid(%) | GSR(ADC) | Cond(µS) | Pitch | Roll | Label        | Time ");
  Serial.println("--------|---------|----------|----------|----------|----------|-------|------|--------------|------");

  sessionStart    = millis();
  lastReadingTime = millis() - READING_INTERVAL_MS; // trigger first reading immediately
  sessionActive   = true;

  blinkLED(3, 150); // 3 quick blinks = session started
}

// ── Loop ─────────────────────────────────────────────────────
void loop() {
  if (!sessionActive) return;

  // ── Continuously check MAX30102 for heartbeat ───────────
  long irValue = particleSensor.getIR();
  if (checkForBeat(irValue)) {
    long delta = millis() - lastBeat;
    lastBeat   = millis();
    beatsPerMinute = 60.0 / (delta / 1000.0);

    if (beatsPerMinute > 20 && beatsPerMinute < 200) {
      rates[rateSpot++] = (byte)beatsPerMinute;
      rateSpot %= RATE_SIZE;
      beatAvg = 0;
      for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
    }
  }

  // ── Time to take a reading? ─────────────────────────────
  long now     = millis();
  long elapsed = now - lastReadingTime;

  if (elapsed >= READING_INTERVAL_MS) {
    lastReadingTime = now;
    readingCount++;

    // Session complete?
    if (readingCount > MAX_READINGS) {
      sessionComplete();
      return;
    }

    // ── Read Sensors ────────────────────────────────────

    // Heart Rate
    long irValue2      = particleSensor.getIR();
    bool fingerDetected = (irValue2 > 50000);
    int  heartRate     = 0;
    if (fingerDetected && beatAvg > 20) {
      heartRate = beatAvg;
    }

    // DHT22 — Temperature & Humidity
    float temp  = dht.readTemperature();
    float humid = dht.readHumidity();
    if (isnan(temp))  temp  = 0.0;
    if (isnan(humid)) humid = 0.0;

    // GSR — Galvanic Skin Response
    int   gsrRaw      = analogRead(GSR_PIN);
    float gsrVolt     = (gsrRaw / 4095.0) * 3.3;
    float conductance = 0.0;
    if (gsrVolt > 0.01) conductance = (1.0 / gsrVolt) * 0.5;
    conductance = round(conductance * 100.0) / 100.0;

    // MPU6050 — Pitch & Roll
    int16_t ax16, ay16, az16, gx16, gy16, gz16;
    mpu.getMotion6(&ax16, &ay16, &az16, &gx16, &gy16, &gz16);
    float axG   = ax16 / 16384.0;
    float ayG   = ay16 / 16384.0;
    float azG   = az16 / 16384.0;
    int   pitch = (int)(atan2(ayG, azG) * 180.0 / PI);
    int   roll  = (int)(atan2(-axG, sqrt(ayG*ayG + azG*azG)) * 180.0 / PI);

    // ── Stress Label ────────────────────────────────────
    String label = determineLabel(heartRate, gsrRaw, temp, humid, fingerDetected);

    // ── Elapsed Time ────────────────────────────────────
    long secElapsed = (millis() - sessionStart) / 1000;
    int  minE = secElapsed / 60;
    int  secE = secElapsed % 60;
    char timeStr[8];
    sprintf(timeStr, "%02d:%02d", minE, secE);

    // ── Serial Table Output ──────────────────────────────
    char line[150];
    sprintf(line,
      "  %4d  | %7d | %8.1f | %8.1f | %8d | %8.2f | %5d | %4d | %-12s | %s",
      readingCount, heartRate, temp, humid,
      gsrRaw, conductance, pitch, roll,
      label.c_str(), timeStr);
    Serial.println(line);

    // ── CSV Output ───────────────────────────────────────
    Serial.print("CSV>>");
    Serial.print(CURRENT_SUBJECT);  Serial.print(",");
    Serial.print(readingCount);     Serial.print(",");
    Serial.print(heartRate);        Serial.print(",");
    Serial.print(temp, 1);          Serial.print(",");
    Serial.print(humid, 0);         Serial.print(",");
    Serial.print(conductance, 2);   Serial.print(",");
    Serial.print(label);            Serial.print(",");
    Serial.print(pitch);            Serial.print(",");
    Serial.println(roll);

    // ── LED Feedback ─────────────────────────────────────
    if      (label == "High_Stress") { blinkLED(3, 80);  }
    else if (label == "Stress")      { blinkLED(2, 100); }
    else if (label == "Normal")      { blinkLED(1, 200); }
    else                             { digitalWrite(ONBOARD_LED, LOW); }

    // ── Progress Update every 50 readings ───────────────
    if (readingCount % 50 == 0) {
      int pct       = (readingCount * 100) / MAX_READINGS;
      int remaining = ((MAX_READINGS - readingCount) * READING_INTERVAL_MS) / 1000;
      Serial.println();
      Serial.print("  ── Progress: ");
      Serial.print(readingCount); Serial.print("/"); Serial.print(MAX_READINGS);
      Serial.print(" ("); Serial.print(pct); Serial.print("%)  —  ");
      Serial.print(remaining / 60); Serial.print("m ");
      Serial.print(remaining % 60); Serial.println("s remaining ──");
      Serial.println();
    }
  }
}

// ── Determine Stress Label ───────────────────────────────────
String determineLabel(int hr, int gsrRaw, float temp, float humid, bool finger) {
  if (!finger || hr == 0) return "No_Contact";

  bool hrHigh  = (hr > HR_STRESS_MAX);
  bool hrMid   = (hr > HR_NORMAL_MAX && hr <= HR_STRESS_MAX);
  bool gsrHigh = (gsrRaw < GSR_STRESS_MIN);
  bool gsrMid  = (gsrRaw >= GSR_STRESS_MIN && gsrRaw < GSR_NORMAL_MIN);
  bool tHigh   = (temp > TEMP_HIGH_MIN);
  bool tMid    = (temp > TEMP_STRESS_MIN && temp <= TEMP_HIGH_MIN);

  int highScore   = (hrHigh  ? 2 : 0) + (gsrHigh ? 2 : 0) + (tHigh ? 1 : 0);
  int stressScore = (hrMid   ? 2 : 0) + (gsrMid  ? 2 : 0) + (tMid  ? 1 : 0);

  if (highScore   >= 3) return "High_Stress";
  if (stressScore >= 3) return "Stress";
  if (highScore   >= 1) return "Stress";
  return "Normal";
}

// ── Session Complete ─────────────────────────────────────────
void sessionComplete() {
  sessionActive = false;
  long totalSec = (millis() - sessionStart) / 1000;

  Serial.println();
  Serial.println("════════════════════════════════════════════════════════");
  Serial.println("  ✅  SESSION COMPLETE!");
  Serial.println("════════════════════════════════════════════════════════");
  Serial.print  ("  Subject      : "); Serial.println(CURRENT_SUBJECT);
  Serial.print  ("  Total Reads  : "); Serial.println(readingCount - 1);
  Serial.print  ("  Total Time   : ");
  Serial.print(totalSec / 60); Serial.print("m ");
  Serial.print(totalSec % 60); Serial.println("s");
  Serial.println();
  Serial.println("  ▶ NEXT STEPS:");
  Serial.println("  1. Copy all CSV>> lines from Serial Monitor");
  Serial.println("  2. Paste into stress_dataset.csv");
  Serial.print  ("  3. Change CURRENT_SUBJECT to ");
  Serial.println(CURRENT_SUBJECT + 1);
  Serial.println("  4. Upload sketch and repeat for next subject");
  Serial.println("════════════════════════════════════════════════════════");

  // Rapid blink to signal done
  for (int i = 0; i < 10; i++) {
    digitalWrite(ONBOARD_LED, HIGH); delay(100);
    digitalWrite(ONBOARD_LED, LOW);  delay(100);
  }
}

// ── Init MAX30102 ────────────────────────────────────────────
void initMAX30102() {
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("  [✗] MAX30102 not found! Check wiring (SDA=21, SCL=22).");
    while (1) { blinkLED(5, 100); delay(1000); }
  }
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeGreen(0);
  Serial.println("  [✓] MAX30102 initialized (I2C 0x57)");
}

// ── Init MPU6050 ─────────────────────────────────────────────
void initMPU6050() {
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("  [!] MPU6050 not found — Pitch/Roll will be 0");
  } else {
    Serial.println("  [✓] MPU6050 initialized (I2C 0x68)");
  }
}

// ── Print Header ─────────────────────────────────────────────
void printHeader() {
  Serial.println();
  Serial.println("════════════════════════════════════════════════════════");
  Serial.println("   AI-IoT Real-Time Stress Detection System");
  Serial.println("   ESP32 | MAX30102 | DHT22 | GSR | MPU-6050");
  Serial.println("   Amity University — CSE Department");
  Serial.println("════════════════════════════════════════════════════════");
  Serial.println("  Initializing sensors...");
}

// ── LED Helper ───────────────────────────────────────────────
void blinkLED(int times, int ms) {
  for (int i = 0; i < times; i++) {
    digitalWrite(ONBOARD_LED, HIGH); delay(ms);
    digitalWrite(ONBOARD_LED, LOW);  delay(ms);
  }
}
