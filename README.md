# IIT Jodhpur NLP Assignment — README

## Overview

This project implements:

* **Problem 1:** Learning Word Embeddings (CBOW & Skip-gram using Word2Vec)
* **Problem 2:** Character-Level Name Generation (RNN, LSTM, Attention)

The code is written in Python and designed to run in a standard environment or Google Colab.

---

## Requirements

Install the required dependencies before running the script:

```bash
pip install gensim beautifulsoup4 requests lxml wordcloud matplotlib scikit-learn nltk torch tqdm
```

---

## Setup

### 1. Clone or Download

Download the repository or copy the Python file:

```
iit_jodhpur_nlp_assignment_fixed_(1).py
```

---

### 2. NLTK Setup

The script automatically downloads required NLTK resources:

* punkt
* stopwords

If needed manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## How to Run

### Option 1: Run Locally

```bash
python iit_jodhpur_nlp_assignment_fixed_(1).py
```

---

### Option 2: Run in Google Colab

1. Open Google Colab
2. Upload the `.py` file
3. Run all cells (Runtime → Run All)

---

## Execution Flow

The script runs sequentially:

### Problem 1: Word Embeddings

1. Scrapes IIT Jodhpur website data
2. Cleans and preprocesses text
3. Trains Word2Vec models (CBOW & Skip-gram)
4. Performs:

   * Nearest neighbor analysis
   * Analogy tasks
   * PCA / t-SNE visualization

---

### Problem 2: Name Generation

1. Loads dataset of Indian names
2. Encodes characters into sequences
3. Trains:

   * Vanilla RNN
   * 2-layer LSTM
   * RNN with Attention
4. Generates new names using temperature sampling
5. Evaluates using:

   * Loss
   * Novelty
   * Diversity

---

## Output

The script produces:

* Trained embedding models
* Generated names
* Evaluation metrics (loss, novelty, diversity)
* Visualizations (PCA, t-SNE, heatmaps)

---

## Notes

* Internet connection is required for web scraping
* Training may take time depending on CPU/GPU
* Default device is CPU


---
