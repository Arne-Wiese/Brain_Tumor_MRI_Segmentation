# Implementation Log: 2.5D U-Net für BraTS2020 Brain Tumor Segmentation

**Datum:** 04.11.2025  
**Ansatz:** 2.5D Convolutional Network mit Slice-Aggregation

## Kernkonzept

- Statt einzelne 2D-Slices → Aggregation von **n=5 konsekutiven Slices** via Mean-Pooling
- Nutzt 3D-Kontext, bleibt aber 2D-Network (effizienter als full 3D)
- Segmentierung erfolgt für den mittleren Slice

## Architektur

- **Model:** Standard U-Net (64→128→256→512→1024)
- **Input:** 4 Kanäle (FLAIR, T1, T1CE, T2), 240×240px
- **Output:** 4 Klassen (Background, NCR, ED, ET)
- **Parameter:** ~31M

## Training Setup

- **Dataset:** BraTS2020 (70% Train, 15% Val, 15% Test)
- **Loss:** CrossEntropyLoss
- **Metrik:** Multi-class Dice Score
- **Optimizer:** Adam (lr=1e-4) + ReduceLROnPlateau
- **Batch Size:** 8
- **Epochs:** 50

## Key Features

- Slice-Range: 2-153 (fokussiert auf tumorhaltige Region)
- Min-Max Normalisierung pro Slice
- Label-Mapping: 4→3 für konsekutive Klassen
- Best model saving basierend auf Val Dice

## Datenpipeline

1. **Slice-Aggregation:** Für jeden Slice werden n/2 Slices davor und danach geladen
2. **Aggregation:** Mittelwert über alle n Slices pro Modalität
3. **Output:** Segmentierungsmaske nur für den mittleren Slice

## Probleme

- Meine Vermutung ist, dass die data pipeline sehr ineffizient ist. Deshalb brauchen wir auch so lange für das Training. Die könnte man wahrscheinlich nochmal anpassen.

## Ergebnisse

_Wird nach Training ergänzt_

- Best Val Dice: TBD
- Test Dice: TBD
- Training Time: TBD
