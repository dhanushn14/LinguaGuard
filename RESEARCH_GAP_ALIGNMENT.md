# Research Gap Alignment: LinguaGuard-FL

This document details how the **LinguaGuard-FL** system directly answers the core research gaps identified in the literature survey of 22 recent papers (2020–2026) conducted at **CARA (Center for Applied Research in AI), RV University**.

---

## 📌 Mapping of Identified Research Gaps to System Implementation

### 1. GAP 2: Low-Resource Language Coverage in Federated NLP
* **Literature Deficit:** Prior works (e.g., Weller et al., NAACL 2022; Huang et al., EMNLP 2020) predominantly benchmarked federated NLP on high-resource English GLUE tasks or major European languages. Dravidian and Indic regional languages have been largely ignored in federated privacy research.
* **LinguaGuard-FL Solution:** Expands beyond initial Kannada-Tamil pilots to incorporate **Malayalam**, **Hindi**, **Tanglish (Tamil-English)**, and **Romanized Kannada** as distinct federated clients, proving that low-resource regional languages can benefit from collaborative model generalization without data sharing.

### 2. GAP 3: Language-Heterogeneous Non-IID Federation
* **Literature Deficit:** Existing federated benchmarks evaluate Non-IID partitioning primarily based on label skew (Dirichlet distribution over classes) or domain/topic differences within the *same* language (Lin et al., FedNLP 2022). Non-IID effects originating from *cross-linguistic morphology, distinct alphabets, and script heterogeneity* remained unstudied.
* **LinguaGuard-FL Solution:** Each client in LinguaGuard-FL represents a linguistically distinct distribution:
  - **Devanagari script** (Hindi) vs. **Malayalam script** (Malayalam) vs. **Latin transliterated script** (Tanglish & Romanized Kannada).
  - Skewed class balance across regional comment sets handled via sample-weighted Federated Averaging (`FedAvg`).

### 3. GAP 4: The "Triple Constraint" (Privacy + Multilingual + Low-Resource)
* **Literature Deficit:** Of the 22 reviewed papers, works tackling multilingual models omitted formal privacy (e.g., Singh & Thakur MultiFED 2024; CARA RP02 Zhao et al. 2024), while works providing Differential Privacy evaluated only English corpora (Mishra & Yadav 2026; Knowledge-Based Systems 2023).
* **LinguaGuard-FL Solution:** Simultaneously provides:
  1. **Federated Learning** (data stays on edge device),
  2. **Adaptive Differential Privacy** (clipping + Gaussian noise perturbation to defend against gradient inversion),
  3. **Lightweight Secure Aggregation Simulation** (masking updates to defend against curious aggregators),
  4. Across **4 low-resource Indian language settings**.

### 4. GAP 7: Code-Switching and Dialectal Social Media Text
* **Literature Deficit:** Social media users in India rarely write in pure literary regional prose; they mix vernacular with English slang in Latin script (e.g., "Padam semma fire bro", "Cinema tumba chennagide"). Existing federated NLP systems assume standard monolingual orthography.
* **LinguaGuard-FL Solution:** Explicitly incorporates **Tanglish** (`vyombhatia/tanglish-comments-for-sentiment-ananlysis`) and **Romanized Kannada** (`sindhur17/romanized-kannada-reviews-dataset`) into the federated ring. The shared subword tokenizer (`distilbert-base-multilingual-cased`) bridges Romanized South Indian vernacular tokens.

### 5. GAP 10: Regulatory Compliance (DPDP Act & GDPR)
* **Literature Deficit:** Standard multilingual NLP pipelines scrape and pool user comments into cloud repositories, exposing personally identifiable information (PII) and violating India's Digital Personal Data Protection (DPDP) Act.
* **LinguaGuard-FL Solution:** Guarantees data locality at the client level. No user comment or review ever leaves the local client boundary; only noisy, bounded, and masked weight updates are communicated.

---

## 🔬 Experimental Comparison Matrix (Thesis Table 5.2 Extended)

| System | Target Languages | Language Families & Scripts | Privacy Mechanisms Applied | Low-Resource Focus |
| :--- | :--- | :--- | :--- | :--- |
| **FedPerX** *(Prakash et al., 2026)* | 10+ Global Languages | European / High-Resource | Adaptive DP on Adapters | Partial (No languages <1K samples) |
| **MultiFED** *(Singh & Thakur, 2024)* | 13 Indic Languages | Indo-Aryan & Dravidian | None (Decentralization only) | Yes (Indic focus, no formal DP) |
| **CARA RP02** *(Zhao et al., ICLR 2024)* | Low-Resource Benchmarks | Diverse global languages | None (Federated Prompt Tuning) | Yes (Prompt tuning focus) |
| **TextHide** *(Huang et al., EMNLP 2020)* | English (GLUE) | Latin / Monolingual | Representation encryption | No (English benchmark only) |
| **LinguaGuard-FL** *(Dhanush N, RVU)* | **Malayalam, Hindi, Tanglish, Kannada** | **Dravidian & Indo-Aryan (Native + Latin Script Code-Mixed)** | **Adaptive DP (Clipping + Calibrated Gaussian Noise) + Secure Aggregation Masking** | **Yes — Dedicated low-resource, code-switched Indian languages** |
