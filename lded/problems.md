# Bekannte Probleme & Fixes

## 1. Tensor-Size-Mismatch: Predicted Heatmaps vs. Target Heatmaps

**Fehler:**
```
RuntimeError: The size of tensor a (256) must match the size of tensor b (128) at non-singleton dimension 3
```

**Ursache:**
Das Modell gibt Heatmaps mit räumlicher Auflösung **256×256** aus, aber das Dataset erzeugt
Target-Heatmaps mit **128×128** (Standard-Default in `model/data/dataset.py`).

**Warum 256×256 vom Modell?**
Der Datenfluss durch das Netz bei Input 512×512:
1. Backbone C3 (Stride 4) → 128×128
2. FPN `target_size` = C3-Größe = 128×128
3. FPN Schritt 4 (`fpn_lite.py` Zeile 159): `out_size = (target_size[0] * 2, target_size[1] * 2)` = **256×256**
4. HeatmapHead erhält 256×256 → gibt 256×256 aus

Das Dataset erzeugt Target-Heatmaps aber mit dem Default `heatmap_size=(128, 128)`.

**Fix:**
`heatmap_size=(256, 256)` an das Dataset übergeben, damit die Targets zur Modell-Ausgabe passen.
Höhere Heatmap-Auflösung ist für präzise Eckpunkt-Lokalisierung vorteilhaft und der
Speicher-Overhead ist bei 4 Kanälen à 256×256 float32 minimal (~1 MB/Batch bei batch_size=32).

**Betroffene Dateien:**
- `scripts/train.py` → `create_dataloaders()`: `heatmap_size=(256, 256)` übergeben
- `model/data/dataset.py` → Default von `(128, 128)` auf `(256, 256)` geändert
- `configs/model/mobilenetv3.yaml` → `heatmap_size` auf `[256, 256]` aktualisiert

## 2. MPS Performance-Degradation: Epoch-Zeiten steigen exponentiell

**Symptom:**
Bei Training auf Apple Silicon (M4 Max) mit `--device mps` steigen die Epoch-Zeiten
drastisch an (9s → 21s → 68s → 69s pro Train-Batch bei nur 50 Samples).

**Ursache:**
Kombination aus drei Faktoren:
1. **MPS Memory Leak** – Ohne explizites `torch.mps.empty_cache()` fragmentiert der
   MPS-Speicher und Allokationen werden zunehmend langsamer.
2. **Albumentations SSL-Check** – Jeder DataLoader-Worker versucht bei jedem Batch einen
   HTTPS-Request für Version-Checks. Bei SSL-Fehler blockiert der Timeout (~5s pro Versuch).
3. **`num_workers > 0` auf macOS/MPS** – Multi-Processing DataLoader auf macOS verursacht
   zusätzlichen Overhead durch Spawning und wiederholte Albumentations-Initialisierung.

**Fix:**
1. `num_workers: 0` in `configs/train/default.yaml` (verhindert Worker-Spawning-Overhead)
2. `torch.mps.empty_cache()` nach jeder Epoch im Training-Loop
3. Umgebungsvariable `ALBUMENTATIONS_CHECK_VERSION=0` vor dem Start setzen

**Erwartete Verbesserung:**
Epoch-Zeiten sollten konstant bei ~2–4s bleiben (50 Samples, batch_size=32) statt exponentiell zu steigen.

**Betroffene Dateien:**
- `configs/train/default.yaml` → `num_workers: 0`
- `scripts/train.py` → `torch.mps.empty_cache()` nach jeder Epoch

## 3. Modell konvergiert zu Null-Heatmaps: PCK=0, NME=95.94 über 100 Epochs

**Symptom:**
Nach 100 Epochs (50 Samples) sinkt der train_loss (3.45 → 0.06), aber PCK@5, PCK@10
und IoU bleiben exakt bei 0.0000. NME bleibt konstant bei 95.9374 – bis auf die 4. 
Nachkommastelle keine Änderung über alle Epochs.

**Diagnose:**
- NME = 95.94% ≈ Fehler wenn alle 4 Ecken bei (0.5, 0.5) vorhergesagt werden
- Soft-Argmax auf einer flachen/Null-Heatmap ergibt immer (0.5, 0.5) (uniformes Gewicht)
- Der AWing-Loss sinkt, weil das Modell lernt überall 0 auszugeben (Ground-Truth-Heatmaps
  sind auch zu ~99.9% Null – nur winzige Gaussian-Peaks mit sigma=3 auf 256×256)

**Ursache:**
1. **sigma=3.0 zu klein für 256×256 Heatmaps** – Peak ist nur ~6px breit auf einem
   65.536px Grid. Das Netz lernt "alles 0" als einfachste Minimierung des Heatmap-Loss.
2. **Kein direkter Koordinaten-Loss** – Der AWing-Loss optimiert nur die Heatmap-Pixel,
   aber das Soft-Argmax-Ergebnis (die eigentlichen Koordinaten) wird nie direkt überwacht.
   Das Netz hat keinen Gradient-Signal das sagt "deine Koordinaten sind falsch".
3. **50 Samples = 1 Batch pro Epoch** – zu wenig Varianz für Generalisierung, aber
   das Hauptproblem ist (1) und (2), da auch mit mehr Daten die gleiche Konvergenz
   zu Null-Heatmaps eintreten würde.

**Fix:**
1. `sigma` von 3.0 auf **10.0** erhöhen → Gaussian-Peaks breiter und leichter zu lernen
2. **Koordinaten-Loss (L1)** als zusätzlichen Loss-Term hinzufügen, der direkt auf die
   Soft-Argmax-Koordinaten wirkt → gibt dem Netz ein klares Signal für die Position
3. `coord_weight: 5.0` als neuer Config-Parameter

**Betroffene Dateien:**
- `model/data/dataset.py` → `sigma` Default von 3.0 auf 10.0
- `model/loss/awing.py` → `LDEDLoss` um L1-Koordinaten-Loss erweitert
- `scripts/train.py` → Koordinaten an den Loss übergeben
- `configs/train/default.yaml` → `coord_weight: 5.0` hinzugefügt

**❌ Ergebnis: Nicht ausreichend.** Nach diesem Fix blieb NME weiterhin konstant bei
95.94, der train_loss schwang wild (3 → 430 → 6 → 230) und val_loss klebte bei ~391.
Die eigentliche Root Cause lag tiefer (siehe Problem 4).

## 4. Architektur-Review: Fehlende Sigmoid + Soft-Argmax-Konflikt (eigentliche Root Cause)

Vollständige Analyse der Architektur inkl. Vergleich mit **DocAligner** (Referenz-Projekt
für Document Edge Detection). Diese Befunde erklären, warum Problem 3 nicht reichte.

### 4.1 Kritisch: Fehlende Sigmoid + Temperature-Problem (Kern-Bug)

Der `heatmap_conv` gab unbegrenzte rohe Logits aus, während der AWing-Loss Werte in
[0, 1] erwartet. Dreifach kaputt:
- **Ohne Sigmoid:** Outputs z.B. −15 bis +15 → AWing-Loss explodiert auf 200–430.
- **Soft-Argmax mit `temperature=1.0`:** Auf 65.536 Pixeln ergibt `e^1 / (65536·e^0) ≈ 0.00004`
  Gewicht am Peak → praktisch uniforme Verteilung.
- **Resultat:** Soft-Argmax gibt immer (0.5, 0.5) zurück → NME konstant 95.94.

**Fix:** `nn.Sigmoid()` am Ende von `heatmap_conv` + `temperature=0.05` im Soft-Argmax
(Literatur-Richtwert: temperature ∈ [0.02, 0.1] bei sigmoid-normalisierten Heatmaps).

### 4.2 Architektonischer Konflikt: AWing vs. Soft-Argmax

AWing optimiert die Heatmap **lokal** (pixelweise Gaussian-Annäherung), Soft-Argmax
normalisiert **global** via Softmax. Inkompatibel ohne scharfen Peak:
- Bimodale Heatmap → Soft-Argmax gibt den Mittelwert zwischen zwei Peaks (kein gültiger Punkt).
- **DocAligner-Lösung:** Kein Soft-Argmax im Inference-Pfad. Stattdessen Heatmap auf
  Originalgröße hochskalieren → Kontur-Analyse → Schwerpunkt des aktivierten Bereichs.

### 4.3 FPN: Kein echtes Top-Down

`fpn_lite.py` skaliert alle Maps (p3, p4_up, p5_up) auf dieselbe Auflösung und addiert –
das ist kein hierarchisches FPN. `f4` und `f5` sind dadurch fast redundant.
Echtes Top-Down: `f3 = fuse(C3, ↑p4)`, `f4 = fuse(C4, ↑p5)`, `p5` unverändert.
**DocAligner** nutzt sogar **BiFPN** (bidirektional, lernbare gewichtete Fusion).

### 4.4 Heatmap-Auflösung 256×256 kontraproduktiv

C5 stammt aus 16×16 – kein echtes Detail über 128×128 hinaus. Das Upsampling auf
256×256 erzeugt nur interpolierte Pixel ohne Mehrwert, verdoppelt aber Speicher/Rechenaufwand.
DocAligner hält die Heatmap-Auflösung bewusst niedrig und löst Präzision per Post-Processing.

### 4.5 Loss-Strategie: Koordinaten-Loss muss Primary sein

AWing trainiert nur die Heatmap-Shape, keine direkte Koordinaten-Supervision.
**DocAligner Phase 1** nutzt Smooth-L1 mit hohem Gewicht direkt auf den Koordinaten.
Neue LDED-Strategie:
```
L_total = 1.0 × L_coord (Smooth-L1 auf Soft-Argmax)
        + 0.3 × L_awing (Auxiliary: Heatmap-Shape)
        + 0.05 × L_bce  (Auxiliary: Confidence)
```

### 4.6 Confidence-Head ohne Negativ-Beispiele

Target ist immer 1.0 → BCE lernt nur eine Konstante. DocAligner setzt auf Negativ-Beispiele
(Bilder ohne Dokument) oder nutzt den Head nur als Deployment-Guard zur Laufzeit.

### Priorisierte Roadmap

| Prio | Änderung | Erwarteter Effekt | Aufwand | Status |
|------|----------|-------------------|---------|--------|
| 🔴 1 | Sigmoid + `temperature=0.05` | NME 95% → <20% | 5 min | ✅ umgesetzt |
| 🔴 2 | Coord-Loss als Primary (Smooth-L1) | Stabile Gradienten | 30 min | ✅ umgesetzt |
| 🔴 3 | LR auf 1e-4, Warmup behalten | Loss-Explosionen enden | 5 min | ✅ umgesetzt |
| 🟡 4 | FPN → echtes Top-Down | Bessere Feature-Fusion | 2h | ⏳ offen |
| 🟡 5 | Confidence-Head entfernen / Negativ-Beispiele | Sinnvoller Loss | 1h | ⏳ offen |
| 🟢 6 | Centroid Post-Processing für Inference | Sub-Pixel-Präzision | 1h | ⏳ offen |
| 🟢 7 | FPN-Lite → BiFPN | +2–5% NME | 4h | ⏳ offen |
| 🟢 8 | Heatmap-Größe auf 128×128 | 4× weniger Speicher | 30 min | ⏳ offen |

**Wichtigster Befund (DocAligner-Vergleich):** DocAligners erste Version (reine
Punkt-Regression) scheiterte aus denselben Gründen. Die Lösung war Heatmap-Regression
mit AWing + **korrektes Post-Processing statt Soft-Argmax im Inference-Pfad** (Centroid-Analyse
auf der hochskalierten Heatmap). Mittelfristig für LDED zu übernehmen.

**Umgesetzte Fixes (Prio 1–3):**
- `model/head/heatmap_head.py` → `nn.Sigmoid()` am Ende von `heatmap_conv`; Soft-Argmax
  `temperature` Default 1.0 → 0.05
- `model/loss/awing.py` → Koordinaten-Loss von L1 auf `SmoothL1Loss` umgestellt
- `configs/train/default.yaml` → `coord_weight: 1.0`, `awing_weight: 0.3`, `bce_weight: 0.05`;
  Lernrate 3e-4 → 1e-4

## 5. Temperatur wird nicht aus der Config durchgereicht (Soft-Argmax bleibt bei 1.0)

**Symptom:**
Obwohl `SoftArgmax2D` und `HeatmapHead` intern den Default `temperature=0.05` hatten,
lief Soft-Argmax beim Training faktisch weiter mit `temperature=1.0` → zu weiche
Verteilung, Koordinaten zur Bildmitte gezogen.

**Ursache:**
- `LDED.__init__` hatte noch `temperature: float = 1.0` als Default.
- `LDED.from_config()` reichte **keinen** Temperatur-Wert weiter.
- Da das Training über `LDED.from_config(...)` baut, landete immer `1.0` im Head und
  überschrieb die korrekten 0.05-Defaults der Unterklassen.
- Zusätzlich fehlte `temperature` komplett im `head`-Block von `mobilenetv3.yaml`.

**Fix:**
- `model/model.py` → `LDED.__init__` Default auf `temperature=0.05`; `from_config()` liest
  `temperature=cfg["head"].get("temperature", 0.05)`.
- `configs/model/mobilenetv3.yaml` → `temperature: 0.05` im `head`-Block ergänzt.

**Betroffene Dateien:**
- `model/model.py`
- `configs/model/mobilenetv3.yaml`

## 6. Warmup-Robustheit + nicht verdrahteter LR-Scheduler

**Symptom / Prüfung:**
Verdacht, der multiplikative Warmup würde die LR über die Epochen kompoundieren
(`pg["lr"] = pg["lr"] * factor`). **Tatsächlich kein aktiver Bug:** Der Optimizer wird in
Phase 1 (Epochs 1–10) jede Epoche neu erzeugt, und `warmup.epochs=5` liegt darin → der
Warmup multipliziert nur einmal pro Epoche auf die frische Base-LR (korrekter linearer Warmup).

**Zwei reale Befunde dabei:**
1. Die Warmup-Logik war **fragil** (würde bei anderen Phasen-/Warmup-Konstellationen
   kompoundieren).
2. Die in `default.yaml` konfigurierte `CosineAnnealingLR` wurde im Code **nie
   instanziiert oder gesteppt** → LR blieb nach dem Warmup konstant.

**Fix:**
- `scripts/train.py` → Pro Param-Gruppe `base_lr` festhalten; Warmup setzt auf `base_lr`
  auf statt zu multiplizieren (robust gegen Optimizer-Neuerstellung).
- LR-Schedule manuell implementiert: linearer Warmup + Cosine-Annealing (aus `base_lr`,
  liest `T_max`/`eta_min` aus Config), pro Epoche neu berechnet → phasen-sicher statt
  optimizer-gebundener `torch`-Scheduler.
- Overfit-Modus: `warmup.epochs=1`, `lr=1e-3`, `T_max` = Overfit-Epochen.

**Betroffene Dateien:**
- `scripts/train.py`

## 7. (Kein Bug) NME-Skalierung in der Validierungs-Pipeline

**Prüfung:**
Vorschlag, in `validate` nicht mit 512 zu multiplizieren und NME auf normalisierten
Koordinaten mit `image_size=(1,1)` zu messen, um eine vermeintliche „NME-Explosion" zu beheben.

**Befund: Änderung wäre kontraproduktiv, kein Bug.**
- Pred und Target werden **konsistent** mit 512 skaliert und durch die Diagonale (724)
  geteilt → mathematisch identisch zu `image_size=(1,1)` (Diagonale √2). Der NME-Wert
  ändert sich also nicht.
- NME liegt strukturell in `[0, ~0.7]` – es gibt keine echte Explosion aus der Metrik.
- `pck()` nutzt **Pixel-Schwellen** (5/10 px). Auf normalisierten Koordinaten `[0,1]`
  wären diese Schwellen immer erfüllt → `PCK@5`/`PCK@10` konstant 1.0 (Metrik kaputt).

**Ergebnis:** Keine Änderung vorgenommen. Die hohen NME-Werte kamen vom Daten-Bug
(siehe Problem 8), nicht von der Metrik.

## 8. Inkonsistente Annotationen: Pixel- statt normalisierte Koordinaten (echte Root Cause des Overfit-Fails)

**Symptom:**
Overfit-Test (5 Samples, 200 Epochs) konvergierte nicht: `train_loss` blieb bei ~259
nahezu konstant (259.77 → 258.11), `NME=271` (unmöglich für `[0,1]`-Koordinaten),
gleichzeitig `PCK@5=0.8` und `IoU=0.78` – scheinbar widersprüchlich.

**Diagnose (empirisch, 1 Batch):**
```
target_corners range: 0.0347 ... 2008.67   ← sollte [0,1] sein!
pred coords range:    0.497 ... 0.502      ← Modell im Bildzentrum festgefahren
loss terms: total=259.7, awing=5.3, bce=0.67, coord=258.1
```
Pro-Sample-Check ergab: `000004.json` speichert `corners` in **Pixel-Koordinaten**
(522…2322) statt normalisiert und hat `image_size: None`. Alle anderen Samples korrekt.

**Ursache:**
Ein einziges fehlerhaftes Sample sprengte über den L1-Coord-Loss den Gesamt-Loss (258)
und NME (271). PCK@5 blieb trotzdem hoch, weil PCK ein reiner Zähler ist (16/20 intakte
Ecken < 5 px) → daher die widersprüchlichen Zahlen. Es lag **nicht** an Temperatur,
Warmup oder Metrik.

**Fix:**
`model/data/dataset.py` → Robustheits-Guard in `__getitem__`: Wenn `corners`-Werte klar
Pixel sind (`np.abs(corners).max() > 1.5`), werden sie anhand der tatsächlichen Bildmaße
normalisiert, bevor Padding/Augmentation laufen.

**Verifikation nach Fix:**
```
target_corners range: 0.0347 ... 0.86   ✅
loss terms: total=1.63, awing=5.2, bce=0.70, coord=0.03   ✅
```

**Offen / Empfehlung:**
Der eigentliche Defekt sitzt in der Annotation-Erzeugung (`data/scripts/`). Der
Dataset-Guard fängt es ab, aber die fehlerhaften JSONs (Pixel-Koordinaten, `image_size: None`)
sollten an der Quelle korrigiert werden.

**Betroffene Dateien:**
- `model/data/dataset.py`
