# 🔬 Research Topics Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=flat-square&logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20NLP-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> A machine learning system that predicts and classifies research topics from academic text (titles, abstracts, or paper content) using Natural Language Processing and classification algorithms.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Results](#results)
- [Enhancements & Roadmap](#enhancements--roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 Overview

This project uses supervised machine learning to automatically predict and classify the research domain or topic of an academic paper based on its textual content. Given a paper's **title** or **abstract**, the model identifies which research area it belongs to (e.g., Computer Vision, Natural Language Processing, Bioinformatics, etc.).

This is useful for:
- Academic search engines to auto-tag papers
- Researchers to explore related literature
- Conference/journal systems for automated paper routing
- Building recommendation systems for research content

---

## ✨ Features

- Text preprocessing pipeline (cleaning, tokenization, stopword removal, stemming/lemmatization)
- Feature extraction using **TF-IDF Vectorization**
- Multi-class classification across research domains
- Model comparison (Logistic Regression, Naive Bayes, Random Forest, SVM)
- Evaluation with accuracy, precision, recall, F1-score
- Prediction on new/unseen paper text

---

## 📁 Project Structure

```
Research_Topics_Prediction_ML/
│
├── data/
│   ├── raw/                    # Original dataset files
│   └── processed/              # Cleaned and preprocessed data
│
├── notebooks/
│   └── Research_Topic_Prediction.ipynb   # Main Jupyter notebook
│
├── models/
│   └── trained_model.pkl       # Saved trained model (after training)
│
├── src/
│   ├── preprocess.py           # Text cleaning & preprocessing
│   ├── features.py             # TF-IDF / feature extraction
│   ├── train.py                # Model training script
│   └── predict.py              # Prediction on new inputs
│
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools/Libraries |
|----------|----------------|
| Language | Python 3.8+ |
| ML Framework | Scikit-learn |
| NLP | NLTK, re, TF-IDF |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed.

### 1. Clone the Repository

```bash
git clone https://github.com/Ashwani4545/Research_Topics_Prediction_ML.git
cd Research_Topics_Prediction_ML
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
```

### 3. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### 4. Run the Notebook

```bash
jupyter notebook notebooks/Research_Topic_Prediction.ipynb
```

---

## ⚙️ How It Works

```
Input Text (Title / Abstract)
         │
         ▼
  Text Preprocessing
  - Lowercasing
  - Remove punctuation & special characters
  - Remove stopwords
  - Stemming / Lemmatization
         │
         ▼
  Feature Extraction
  - TF-IDF Vectorization
         │
         ▼
  ML Classification Model
  - Logistic Regression / SVM / Random Forest
         │
         ▼
  Predicted Research Topic / Domain
```

---

## 📊 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~85% |
| Naive Bayes | ~80% |
| Support Vector Machine | ~87% |
| Random Forest | ~83% |

> Note: Results may vary based on dataset size and quality.

---

## 🚀 Enhancements & Roadmap

This section outlines how to evolve this from a notebook experiment into a **production-ready, portfolio-worthy project**.

### ✅ Phase 1 — Strengthen the Core ML (Immediate)

- [ ] **Use a larger, richer dataset** — Scrape papers from [arXiv](https://arxiv.org) or use the [Semantic Scholar Open Research Corpus](https://api.semanticscholar.org) via their free API
- [ ] **Add cross-validation** — Use `StratifiedKFold` instead of a single train-test split for more reliable accuracy metrics
- [ ] **Hyperparameter tuning** — Use `GridSearchCV` or `RandomizedSearchCV` on your best-performing model
- [ ] **Confusion matrix & classification report** — Add detailed per-class performance metrics and visualize them with `seaborn.heatmap`

### 🧠 Phase 2 — Better NLP & Embeddings (Intermediate)

- [ ] **Replace TF-IDF with Word Embeddings** — Use `Word2Vec`, `GloVe`, or `FastText` for richer semantic features
- [ ] **Use pretrained Transformers** — Fine-tune `BERT` or `SciBERT` (a BERT variant trained on scientific text) using HuggingFace Transformers for a significant accuracy boost
- [ ] **Add multi-label classification** — Many papers span multiple topics; extend the model to predict multiple relevant domains
- [ ] **Keyword/Keyphrase extraction** — Integrate `KeyBERT` or `YAKE` to extract top keywords from predictions

### 🌐 Phase 3 — Build a Web App (Deployment)

- [ ] **Flask / FastAPI backend** — Wrap the trained model in a REST API endpoint (`POST /predict`)
- [ ] **Streamlit frontend** — Build an interactive UI where users paste an abstract and get a predicted topic instantly
- [ ] **Docker** — Containerize the app with a `Dockerfile` for portable, reproducible deployment
- [ ] **Deploy to cloud** — Host on **Hugging Face Spaces** (free), **Render**, or **Railway** for a live demo link on your GitHub

### 📦 Phase 4 — Engineering & Production Quality

- [ ] **MLflow experiment tracking** — Log all runs, hyperparameters, and metrics to compare experiments
- [ ] **Model versioning with DVC** — Track your datasets and model files like code using `dvc`
- [ ] **Automated retraining pipeline** — Build a script that fetches new papers and retrains the model on a schedule
- [ ] **Unit tests** — Add `pytest` tests for your preprocessing functions and prediction pipeline
- [ ] **CI/CD with GitHub Actions** — Auto-run tests on every push

### 📄 Phase 5 — Documentation & Presentation

- [ ] **Add a demo GIF/screenshot** to the README showing the app in action
- [ ] **Write a blog post** on Medium or Dev.to explaining your approach — great for visibility
- [ ] **Add a `CONTRIBUTING.md`** to invite collaborators
- [ ] **Dataset card** — Document your dataset (source, size, class distribution) in a `data/README.md`

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: your feature description"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please make sure your code follows PEP8 style and includes comments where necessary.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Ashwani**  
GitHub: [@Ashwani4545](https://github.com/Ashwani4545)

---

> ⭐ If you found this project useful, please give it a star — it helps others discover it!
