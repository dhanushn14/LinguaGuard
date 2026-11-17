# LinguaGuard-FL: Privacy-Preserving Multilingual Sentiment Analysis using Federated Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flower FL](https://img.shields.io/badge/Flower-1.4+-ff69b4.svg)](https://flower.ai/)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/)

**Author:** Dhanush N (USN: `1RUA24CSE0123`)  
**Supervisor / Guide:** Prof. Rahul Panja  
**Department / Center:** School of Computer Science and Engineering, CARA (Center for Applied Research in AI), RV University, Bengaluru  

---

## 📖 1. Project Overview & Motivation

Sentiment analysis systems for low-resource Indian languages conventionally centralize private user data (social media comments, personal reviews) onto a single server. This central pooling creates major data privacy risks and violates statutory regulations such as the **European GDPR** and India's **Digital Personal Data Protection (DPDP) Act**.

**LinguaGuard-FL** solves this through a decentralized, privacy-preserving paradigm:
- **Federated Learning (FL):** Language-specific clients fine-tune a shared multilingual transformer backbone (`distilbert-base-multilingual-cased`) locally without transmitting raw text.
- **Adaptive Differential Privacy (ADP):** Clips local parameter updates ($[-C, C]$) and injects calibrated zero-mean Gaussian noise ($\sigma = 0.001$) to defend against model/gradient inversion attacks.
- **Lightweight Secure Aggregation Simulation:** Perturbs privatized weights with randomized masking noise ($\sigma_{mask} = 0.0005$) prior to transmission, demonstrating cryptographic confidentiality against curious aggregators.
- **Federated Averaging (FedAvg):** Coordinated by a Flower framework server that performs dataset-size-weighted aggregation across rounds.

---

## 🎯 2. The Four Language Datasets

This project scales from 2 clients (Kannada & Tamil) to a diverse 4-client federation covering both **Dravidian** and **Indo-Aryan** language families, including **native scripts**, **code-switching**, and **Romanized transliteration**:

| Client # | Language | Script / Modality | Kaggle Dataset Slug | Sentiment Classes |
| :---: | :---: | :---: | :---: | :---: |
| **Client 1** | **Malayalam** | Native Malayalam Script | `abhijith7711/malayalam-movie-review-dataset` | Negative (0), Neutral/Mixed (1), Positive (2) |
| **Client 2** | **Hindi** | Devanagari Script | `disisbig/hindi-movie-reviews-dataset` | Negative (0), Neutral/Mixed (1), Positive (2) |
| **Client 3** | **Tanglish** | Code-Mixed Tamil + English (Latin Script) | `vyombhatia/tanglish-comments-for-sentiment-ananlysis` | Negative (0), Neutral/Mixed (1), Positive (2) |
| **Client 4** | **Kannada** | Romanized Kannada (Latin Script) | `sindhur17/romanized-kannada-reviews-dataset` | Negative (0), Neutral/Mixed (1), Positive (2) |

---

## 🏗️ 3. Repository Architecture

```text
LinguaGuard-FL/
├── configs/
│   └── config.py                     # Central configuration (hyperparameters, DP budgets, paths)
├── data/
│   ├── dataset_loader.py             # PyTorch SentimentDataset & DataLoader builders
│   ├── kannada/                      # train.csv, test.csv (Romanized Kannada reviews)
│   ├── tamil/                        # train.csv, test.csv (Tanglish / Code-switched Tamil)
│   ├── malayalam/                    # train.csv, test.csv (Native Malayalam reviews)
│   └── hindi/                        # train.csv, test.csv (Devanagari Hindi reviews)
├── models/
│   └── distilbert_model.py           # DistilBERT wrapper & parameter extraction/loading
├── privacy/
│   ├── adaptive_dp.py                # Adaptive Differential Privacy (clipping + Gaussian noise)
│   └── secure_aggregation.py         # Secure Aggregation simulation (random mask perturbation)
├── training/
│   └── trainer.py                    # Local client training engine & evaluation loops
├── clients/
│   └── client.py                     # FlowerClient (NumPyClient) wrapping trainer & privacy modules
├── server/
│   └── server.py                     # Central Flower server with FedAvg & weighted metrics aggregation
├── scripts/
│   ├── download_datasets.py          # Kaggle API downloader + offline synthetic seed generator
│   ├── clean_dataset.py              # Schema-agnostic cleaning & 3-class label normalizer
│   ├── run_federated_simulation.py   # In-process 4-client federated simulation runner
│   └── run_centralized_baseline.py   # Pooled non-private centralized baseline benchmark
├── utils/
│   └── metrics.py                    # Accuracy, precision, recall, F1, confusion matrix, logging
├── notebooks/
│   ├── LinguaGuard_FL_Multilingual_Training.ipynb  # Interactive Colab & Kaggle notebook
│   └── build_notebook.py                           # Python builder to generate the notebook
├── datasets/
│   ├── raw/                          # Raw CSV files
│   └── processed/                    # Cleaned 2-column CSV files (text, label)
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 🚀 4. How to Run the Notebook (Google Colab & Kaggle)

The notebook `notebooks/LinguaGuard_FL_Multilingual_Training.ipynb` is an all-in-one file ready to execute directly on **Google Colab** or **Kaggle Notebooks**.

### Running in Google Colab:
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File > Upload Notebook** and upload `LinguaGuard_FL_Multilingual_Training.ipynb`.
3. Switch runtime to GPU: **Runtime > Change runtime type > T4 GPU**.
4. Run cells sequentially:
   - Cell 1 installs dependencies (`transformers`, `flwr`, `datasets`, `scikit-learn`, `seaborn`).
   - Cell 4 automatically downloads datasets (via Kaggle API or uses built-in seed datasets if no key is configured).
   - Cell 10 trains the **Centralized Baseline** as reference.
   - Cell 11 runs the **4-Client Federated Simulation** with Adaptive DP & Secure Aggregation.
   - Cells 12-14 generate distributed loss/accuracy curves, per-language confusion matrices, and the privacy-utility tradeoff table.

### Running in Kaggle Notebooks:
1. Create a new Notebook on Kaggle.
2. Select **Settings > Accelerator > GPU P100** or **GPU T4 x2**.
3. Under **Add Input / Add Data**, search and add:
   - `abhijith7711/malayalam-movie-review-dataset`
   - `disisbig/hindi-movie-reviews-dataset`
   - `vyombhatia/tanglish-comments-for-sentiment-ananlysis`
   - `sindhur17/romanized-kannada-reviews-dataset`
4. Upload `LinguaGuard_FL_Multilingual_Training.ipynb` and click **Run All**. The notebook automatically checks `/kaggle/input` and locates the datasets.

---

## 💻 5. Running Locally (VS Code / Terminal)

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Download and Clean Datasets
```bash
# Downloads raw data from Kaggle (or creates seed sets if offline)
python scripts/download_datasets.py

# Normalizes labels to [0: Negative, 1: Neutral/Mixed, 2: Positive]
python scripts/clean_dataset.py
```

### Step 3: Run Centralized Baseline (Benchmark)
```bash
python scripts/run_centralized_baseline.py
```

### Step 4: Run Federated Learning (Single-Command Simulation)
Runs all 4 clients concurrently across 3 rounds with FedAvg, Adaptive DP, and Secure Aggregation:
```bash
python scripts/run_federated_simulation.py
```

### Step 5 (Alternative): Run Multi-Terminal Distributed Architecture
To simulate separate network nodes (matching the thesis architecture):
- **Terminal 1 (Server):**
  ```bash
  python server/server.py --rounds 3 --min_clients 4
  ```
- **Terminal 2 (Malayalam Client):**
  ```bash
  python clients/client.py --language malayalam
  ```
- **Terminal 3 (Hindi Client):**
  ```bash
  python clients/client.py --language hindi
  ```
- **Terminal 4 (Tanglish Client):**
  ```bash
  python clients/client.py --language tanglish
  ```
- **Terminal 5 (Kannada Client):**
  ```bash
  python clients/client.py --language kannada
  ```

---

## 📊 6. Theoretical & Empirical Summary

- **Backbone Architecture:** `distilbert-base-multilingual-cased` (134M parameters, 104 languages, distilled 6-layer Transformer).
- **Federated Aggregation:** FedAvg with sample-count weighting:
  $$W_{global}^{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} W_k^{t+1}$$
- **Adaptive Differential Privacy:**
  $$W_{DP} = \text{clip}(W, -C, C) + \mathcal{N}(0, \sigma^2 \mathbf{I})$$
  with clipping bound $C = 1.0$ and noise multiplier $\sigma = 0.001$.
- **Secure Aggregation Simulation Mask:**
  $$W_{SecAgg} = W_{DP} + \mathcal{N}(0, \sigma_{mask}^2 \mathbf{I})$$
  with mask scale $\sigma_{mask} = 0.0005$.
