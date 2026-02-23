# Hotel Review Sentiment Analysis | LSTM NLP

LSTM-based deep learning model for 3-class sentiment classification (Negative / Neutral / Positive) of hotel customer reviews.

## Problem Statement
Automatically classify hotel reviews into **negative, neutral, or positive** sentiment categories based on review text and star ratings, enabling automated feedback analysis at scale.

## Dataset
| Attribute | Detail |
|---|---|
| File | `dataset.csv` |
| Records | ~20,475 English hotel reviews (after language filtering) |
| Features | `Review` (text), `Rating` (1–5 stars) |
| Target | Sentiment: Negative (1–2★) / Neutral (3★) / Positive (4–5★) |
| Class Balance | ~73.7% Positive / ~15.7% Negative / ~10.6% Neutral |

## Methodology
1. **Language Detection** — Filter non-English reviews using `langdetect`
2. **Text Preprocessing** — Lowercasing, special character removal, tokenization, stopword filtering, lemmatization (NLTK)
3. **Class Balancing** — `RandomOverSampler` (neutral ↑ to 5,000) + `RandomUnderSampler` (positive ↓ to 10,000)
4. **Sequence Encoding** — `Tokenizer` (max 10,000 words) + `pad_sequences` (max_len=150)
5. **LSTM Model** — `Embedding(10k, 128)` + `SpatialDropout1D` + `LSTM(128)` + `Dense(3, softmax)`
6. **Training** — 5 epochs, batch size 64, `EarlyStopping` (patience=3)
7. **Evaluation** — Accuracy, Precision, Recall, F1-Score per class

## Results
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Negative | 0.68 | 0.56 | 0.61 |
| Neutral | 0.25 | 0.35 | 0.29 |
| **Positive** | **0.90** | **0.88** | **0.89** |
| **Overall Accuracy** | — | — | **0.77** |

> Neutral class (F1=0.29) is the weakest — ambiguous 3-star reviews are inherently difficult to classify.

## Technologies
`Python` · `TensorFlow/Keras` · `NLTK` · `langdetect` · `imbalanced-learn` · `Pandas` · `NumPy` · `scikit-learn` · `joblib`

## File Structure
```
20_Hotel_Review_NLP/
├── projct_notebook.ipynb         # Main notebook
├── dataset.csv                   # Hotel review dataset
├── src/
│   └── data_preprocessing.py    # Preprocessing utilities
└── data/processed/
    ├── sentiment_model_dl.h5     # Saved LSTM model
    └── tokenizer_dl.pkl          # Saved tokenizer
```

## How to Run
```bash
cd 20_Hotel_Review_NLP
jupyter notebook projct_notebook.ipynb
```
