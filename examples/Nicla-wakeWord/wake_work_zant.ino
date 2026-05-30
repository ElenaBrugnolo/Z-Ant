#include <PDM.h>
#include <lib_zant.h>

// --- DICHIARAZIONI Z-ANT ---
extern "C" void zant_timeseries_to_image(const float *signal, float *output);

#define SIGNAL_LEN 64
#define IMAGE_SIZE 64
#define OUT_LEN 2      // Numero di classi (es. 0: Rumore, 1: Wake Word)

// Allocazione statica (Zero heap) come nel paper
static float timeseriesData[SIGNAL_LEN];
static float imageData[3 * IMAGE_SIZE * IMAGE_SIZE];
static uint32_t inputShape[4] = {1, 3, IMAGE_SIZE, IMAGE_SIZE};

// --- CONFIGURAZIONE AUDIO ---
#define SAMPLE_RATE 16000
#define SAMPLES_PER_WINDOW 250 // 16000 campioni / 64 feature = 250

short sampleBuffer[512]; // Buffer temporaneo per i dati PDM grezzi
volatile int featureIndex = 0;
volatile int sampleAccumulatorCount = 0;
volatile float sumOfSquares = 0.0;
volatile bool readyForInference = false;

// --- FUNZIONI QSPI (Mantenute dal paper originale) ---
#include "stm32h7xx_hal.h"
extern QSPI_HandleTypeDef hqspi;

extern "C" HAL_StatusTypeDef qspi_init_16mb(QSPI_HandleTypeDef *hqspi);
extern "C" HAL_StatusTypeDef enable_quad(QSPI_HandleTypeDef *hqspi);
extern "C" HAL_StatusTypeDef qspi_enter_mmap(QSPI_HandleTypeDef *hqspi);

// --- INTERRUPT AUDIO (Callback PDM) ---
void onPDMdata() {
  // Leggi i dati disponibili dal microfono
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  int samplesRead = bytesAvailable / 2; // I campioni PCM sono a 16 bit (2 byte)

  // Se stiamo già facendo inferenza, scarta i nuovi campioni per non sovrascrivere
  if (readyForInference) return;

  for (int i = 0; i < samplesRead; i++) {
    float val = (float)sampleBuffer[i];
    sumOfSquares += (val * val);
    sampleAccumulatorCount++;

    // Quando abbiamo accumulato un "cesto" da 250 campioni...
    if (sampleAccumulatorCount >= SAMPLES_PER_WINDOW) {
      // Calcola l'RMS
      float rms = sqrt(sumOfSquares / SAMPLES_PER_WINDOW);
      timeseriesData[featureIndex++] = rms;
      
      // Resetta per il prossimo cesto
      sumOfSquares = 0.0;
      sampleAccumulatorCount = 0;

      // Se abbiamo riempito tutte e 64 le feature (è passato 1 secondo)
      if (featureIndex >= SIGNAL_LEN) {
        readyForInference = true;
        featureIndex = 0;
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(5000); // Attendi per aprire il serial monitor

  // 1. INIZIALIZZAZIONE QSPI (Identica al paper)
  if (qspi_init_16mb(&hqspi) != HAL_OK) {
    Serial.println("QSPI init FAIL"); for (;;) {}
  }
  if (enable_quad(&hqspi) != HAL_OK) {
    Serial.println("Enable QE FAIL"); for (;;) {}
  }
  if (qspi_enter_mmap(&hqspi) != HAL_OK) {
    Serial.println("XIP FAIL"); for (;;) {}
  }
  Serial.println("QSPI Ready. Pesi della rete mappati.");

  // 2. INIZIALIZZAZIONE MICROFONO PDM
  PDM.onReceive(onPDMdata);
  // Usa il canale 1 (mono), a 16 kHz
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Errore avvio PDM!");
    while (1);
  }
  Serial.println("Microfono in ascolto...");
}

void loop() {
  // Attendi che la callback audio abbia estratto 64 feature RMS
  if (readyForInference) {
    unsigned long t_start = millis();

    // 1. CODIFICA IMMAGINE (GASF/GADF/MTF in Zig)
    zant_timeseries_to_image(timeseriesData, imageData);
    unsigned long t_img = millis();

    // 2. INFERENZA Z-ANT
    float *out = nullptr;
    int rc = predict(imageData, inputShape, 4, &out);
    unsigned long t_inf = millis();

    // 3. ANALISI RISULTATO
    if (rc == 0 && out) {
      int best = 0;
      for (int i = 1; i < OUT_LEN; i++) {
        if (out[i] > out[best]) best = i;
      }

      Serial.print("Classe predetta: "); 
      Serial.print(best);
      Serial.print(" | Confidenza: ");
      
      // Stampa i punteggi per le classi
      for (int i = 0; i < OUT_LEN; i++) {
        Serial.print(out[i], 4);
        if (i < OUT_LEN - 1) Serial.print(" / ");
      }
      Serial.println();

      // Logica Trigger Wake Word (Supponendo Classe 1 = Wake Word)
      if (best == 1 && out[1] > 0.80) {
          Serial.println(">>> HEY FRANCO RILEVATO! <<<");
      }

    } else {
      Serial.println("Errore di inferenza Z-ANT");
    }

    Serial.print("Tempi -> Encoding: "); Serial.print(t_img - t_start);
    Serial.print("ms | Inferenza: "); Serial.print(t_inf - t_img);
    Serial.println("ms\n");

    // Sblocca il flag per permettere al microfono di ascoltare il prossimo secondo
    readyForInference = false;
  }
}