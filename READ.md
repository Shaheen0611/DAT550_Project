# 📰 Hyperpartisan News Detection: Traditional ML and Fine-Tuned BERT

This repository contains the implementation and experiments for detecting hyperpartisan news articles using both traditional machine learning techniques and transformer-based deep learning (BERT). The project was conducted as part of a university research effort aligned with **SemEval-2019 Task 4: Hyperpartisan News Detection**.

---

## 📄 Project Summary

The goal of this project is to classify whether a news article exhibits **hyperpartisan bias** (i.e., strong ideological leaning) using full article text. Two approaches were developed and compared:

1. **Traditional Machine Learning Pipeline**:
   - Features: TF-IDF + Stylistic Features
   - Models: Logistic Regression, Random Forest

2. **Deep Learning Approach**:
   - Fine-tuned `bert-base-uncased` using Hugging Face Transformers
   - Utilizes contextual embeddings and sequence classification

Results are visualized using PCA, ROC curves, and confusion matrices.

---

## 🗂 Project Structure

