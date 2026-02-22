# STG-Mixer

A PyTorch Lightning implementation of **STG-Mixer** for **hand, foot, and mouth disease (HFMD)** forecasting with spatio-temporal graph learning.

---

## Title
**STG-Mixer** — Spatio-temporal Graph Mixer for HFMD prediction.

---

## Description
This repository contains code to train and evaluate an STG-Mixer model for HFMD prediction. The model leverages a **graph structure** (adjacency matrix) to capture spatial dependencies among regions and uses **time series inputs** that combine HFMD cases with **exogenous signals** (meteorology and weekly/weekend patterns). Training, checkpointing, and evaluation are implemented with **PyTorch Lightning**.

---

## Dataset Information (Overview)
The project uses **four CSV files** (all read **without headers** and cast to `float32`), aligned on the same timeline and node order:

1. **HFMD case series**: `data/hfmd_num1.csv`  
   Multivariate time series of HFMD values over time for **47 nodes/regions**.

2. **Adjacency matrix**: `data/adj0.csv`  
   A **47×47** matrix describing spatial relationships between nodes (binary or weighted), used as the graph structure.

3. **Meteorological series**: `weather.csv` *(currently loaded via an absolute path in code)*  
   Time-aligned weather variables used as exogenous features. The current implementation selects **3 weather features per node** using a fixed column-selection rule.

4. **Weekly/Weekend series**: `weekend.csv` *(currently loaded via an absolute path in code)*  
   Time-aligned calendar/weekly signal (per node) appended as an additional exogenous feature.

> Note: In `functions.py`, the weather and weekend files are read from:
> - `/home/yn/cx/wzk/yuce/data/weather.csv`
> - `/home/yn/cx/wzk/yuce/data/weekend.csv`

---

## Code Information
Key components:

- **Entry script**: `main.py`  
  - Builds the data module, model, and training task from command-line arguments  
  - Uses callbacks (early stopping + checkpointing)  
  - Runs `trainer.fit(...)` then automatically runs `trainer.test(..., ckpt_path="best")`

- **Model**: `models.STGMixer`  
  Instantiated with `adj`, `hidden_dim`, `seq_len`, and `pre_len`.

- **Task**: `tasks.SupervisedForecastTask`  
  LightningModule that defines training/validation/testing steps and metrics.  
  Validation monitoring is set to `val_r2`.

- **Data**: `utils.data.SpatioTemporalCSVDataModule`  
  Loads `hfmd_num1.csv` and `adj0.csv` via `DATA_PATHS`, prepares dataloaders.

- **Utilities / preprocessing**: `functions.py`  
  Includes CSV loading, min-max normalization, feature fusion (case + weather + weekend), and sliding-window dataset generation.

---

## Usage Instructions

### 1) Prepare files
Ensure these files exist:

- `data/hfmd_num1.csv` (HFMD cases)
- `data/adj0.csv` (adjacency matrix)
- `data/weather.csv` (meteorology)
- `data/weekend.csv` (weekly/weekend signal)

### 2) Run experiments

```bash
python main.py --pre_len 1 --batch_size 16
python main.py --pre_len 2 --batch_size 16
python main.py --pre_len 3 --batch_size 16

## Requirements

This project was run in a Conda environment (Python 3.9). Core dependencies include:
python==3.9.23
pytorch==1.12.0 (CUDA 11.3)
pytorch-lightning==1.5.9
numpy==1.26.4
pandas==2.2.3
matplotlib==3.9.4
scikit-learn==1.6.1
scipy==1.13.1
torchmetrics==1.5.2
tqdm==4.67.1
CUDA-related packages (for GPU training):
cudatoolkit==11.3.1
torchvision==0.13.0
torchaudio==0.12.0
