# Hyperpartisan News Detection: Traditional ML and Fine-Tuned BERT

This repository implements two classification pipelines to detect hyperpartisan news articles using both traditional machine learning techniques and a transformer-based deep learning approach (BERT). The results are based on experiments described in the project report titled:

> **"Hyperpartisan News Detection Using Traditional Machine Learning and Fine-Tuned BERT"**  
> _Shaheen Thayalan, University of Stavanger, 2025_

---

## Project Summary

The task involves identifying hyperpartisan news articles (strong ideological bias) using full-text news content. Two approaches are implemented:

1. **Traditional Machine Learning**
   - Features: TF-IDF and stylistic/sentiment-based features
   - Models: Logistic Regression, Random Forest

2. **Transformer-based Deep Learning**
   - Model: `bert-base-uncased`
   - Framework: Hugging Face Transformers

Visualizations include PCA projections, ROC curves, and confusion matrices. Misclassified samples are also reviewed.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Download required NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

---

## Reproducing Results

### Traditional ML Pipeline (TF-IDF + Stylistic Features)

Make sure the following XML files are present in the root directory:

- `articles-training-byarticle-20181122.xml`
- `ground-truth-training-byarticle-20181122.xml`
- `articles-test-byarticle-20181207.xml`
- `ground-truth-test-byarticle-20181207.xml`

Open `main.py` (or `Hyperpartisan News Detection.py`) and ensure:

```python
USE_BERT = False
```

Then run:

```bash
python "Hyperpartisan News Detection.py"
```

This pipeline performs:

- XML parsing and text cleaning
- TF-IDF vectorization (10,000 features)
- Stylistic feature extraction:
  - Readability: Flesch-Kincaid, Gunning Fog
  - Sentiment: polarity, subjectivity
  - POS tags: adjective/adverb counts
  - Lexical diversity, avg. sentence/word length
- PCA visualization
- Train & evaluate Logistic Regression and Random Forest
- Plot ROC curve
- Save models as `.joblib`

### BERT-Based Classifier

Open `main.py` and set:

```python
USE_BERT = True
```

Run:

```bash
python "Hyperpartisan News Detection.py"
```

This will:

- Tokenize with `BertTokenizerFast`
- Prepare datasets using `datasets.Dataset`
- Fine-tune `bert-base-uncased` using Hugging Face Trainer
- Evaluate on test set with classification report
- Save confusion matrix (`Figure_BERT_Confusion.png`)
- Save model + tokenizer to `bert_model/`

> **Note:** Training BERT on CPU is slow. Use GPU if available.

---

## Outputs

| File                        | Description                                |
|----------------------------|--------------------------------------------|
| `logistic_model.joblib`    | Trained Logistic Regression model          |
| `randomforest_model.joblib`| Trained Random Forest model                |
| `tfidf_vectorizer.joblib`  | TF-IDF vectorizer fitted on training data  |
| `bert_model/`              | Saved fine-tuned BERT model + tokenizer    |
| `Figure_BERT_Confusion.png`| Confusion matrix plot for BERT             |
| `Figure_1.png` - `Figure_3.png` | PCA, ROC, and other visualizations   |

---

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC AUC (for Logistic Regression)
- PCA for visualizing feature space

---

## Project Structure

```
.
├── articles-training-byarticle-20181122.xml
├── ground-truth-training-byarticle-20181122.xml
├── articles-test-byarticle-20181207.xml
├── ground-truth-test-byarticle-20181207.xml
├── main.py / Hyperpartisan News Detection.py
├── requirements.txt
├── README.md
├── *.joblib
├── Figure_*.png
├── bert_model/
└── kiesel_2019c.pdf
```

---

## References

- **SemEval-2019 Task 4: Hyperpartisan News Detection**  
  Kiesel, J., Mestre, M., Shukla, R., Vincent, E., et al. (2019)  
  [PDF included: `kiesel_2019c.pdf`]

- **Libraries**: 
  - Hugging Face Transformers: https://github.com/huggingface/transformers
  - NLTK, TextBlob, TextStat for feature extraction

---

## Author

**Shaheen Thayalan**  
**280061@uis.no**  
**University of Stavanger**


