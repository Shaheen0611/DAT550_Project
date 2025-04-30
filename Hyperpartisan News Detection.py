import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import nltk
from textblob import TextBlob
import textstat
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from scipy.sparse import hstack
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Use BERT model instead of TF-IDF + Logistic Regression (set True to enable)
USE_BERT = False

# File paths
TRAIN_ARTICLES_FILE = 'articles-training-byarticle-20181122.xml'
TRAIN_LABELS_FILE = 'ground-truth-training-byarticle-20181122.xml'
TEST_ARTICLES_FILE = 'articles-test-byarticle-20181207.xml'
TEST_LABELS_FILE = 'ground-truth-test-byarticle-20181207.xml'

# Download required NLTK models
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Parse articles from XML
def parse_articles(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    articles = []
    for article in root.findall('article'):
        aid = article.get('id')
        title = article.get('title') or ''
        text = ''.join(article.itertext())
        articles.append({'id': aid, 'title': title, 'text': text})
    return pd.DataFrame(articles)

# Parse labels from XML
def parse_labels(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = [{'id': article.get('id'), 'label': 1 if article.get('hyperpartisan') == 'true' else 0}
              for article in root.findall('article')]
    return pd.DataFrame(labels)

# Clean unwanted text patterns
def clean_text(text):
    text = re.sub(r'Help \| Press \| Advertise.*?All rights reserved\.', '', text, flags=re.DOTALL)
    text = re.sub(r'Site Map.*?All rights reserved\.', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Extract style-based features from text
def extract_stylistic_features(texts):
    features = []
    for text in texts:
        blob = TextBlob(text)
        tokens = blob.words
        sents = blob.sentences
        pos_tags = blob.tags

        word_count = len(tokens)
        sent_count = len(sents)
        avg_word_len = np.mean([len(w) for w in tokens]) if word_count else 0
        avg_sent_len = word_count / sent_count if sent_count else 0
        lexical_diversity = len(set(tokens)) / word_count if word_count else 0

        pos_counts = nltk.FreqDist(tag for (_, tag) in pos_tags)
        adj_count = pos_counts['JJ'] + pos_counts['JJR'] + pos_counts['JJS']
        adv_count = pos_counts['RB'] + pos_counts['RBR'] + pos_counts['RBS']

        features.append([
            textstat.flesch_kincaid_grade(text),
            textstat.gunning_fog(text),
            blob.sentiment.polarity,
            blob.sentiment.subjectivity,
            lexical_diversity,
            avg_word_len,
            avg_sent_len,
            adj_count,
            adv_count
        ])
    return np.array(features)

# Load and prepare the data
print("Loading data...")
train_df = parse_articles(TRAIN_ARTICLES_FILE)
train_labels_df = parse_labels(TRAIN_LABELS_FILE)
train_df = train_df.merge(train_labels_df, on='id')

test_df = parse_articles(TEST_ARTICLES_FILE)
test_labels_df = parse_labels(TEST_LABELS_FILE)
test_df = test_df.merge(test_labels_df, on='id')

# Clean article text
train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

# Use TF-IDF and logistic regression unless BERT is enabled
if not USE_BERT:
    print("Extracting features...")
    style_train = extract_stylistic_features(train_df['text'])
    style_test = extract_stylistic_features(test_df['text'])

    print("Using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(train_df['text'])
    X_test_tfidf = vectorizer.transform(test_df['text'])

    # Combine TF-IDF and stylistic features
    X_train_combined = hstack([X_train_tfidf, style_train])
    X_test_combined = hstack([X_test_tfidf, style_test])

    y_train = train_df['label']
    y_test = test_df['label']

    # Visualize data with PCA
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(X_train_combined.toarray())
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced_X[:,0], reduced_X[:,1], c=y_train, cmap='coolwarm', alpha=0.5)
    plt.title('PCA Projection of Articles')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes", labels=["Not Hyperpartisan", "Hyperpartisan"])
    plt.gca().add_artist(legend1)
    plt.show()

    # Train and evaluate logistic regression
    print("Training Logistic Regression...")
    clf = LogisticRegression(C=0.5, max_iter=1000)
    clf.fit(X_train_combined, y_train)

    print("Evaluating Logistic Regression...")
    y_pred = clf.predict(X_test_combined)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred))

    # Train and evaluate Random Forest
    print("Training Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    rf_clf.fit(X_train_combined, y_train)

    print("Evaluating Random Forest...")
    y_pred_rf = rf_clf.predict(X_test_combined)
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Report:\n", classification_report(y_test, y_pred_rf))

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test_combined)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Save models
    joblib.dump(clf, "logistic_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    joblib.dump(rf_clf, "randomforest_model.joblib")

    # Show misclassified articles
    print("\n- Misclassified Articles -")
    misclassified = test_df[y_test != y_pred]
    for idx, row in misclassified.iterrows():
        print(f"\n[ID: {row['id']}] Predicted: {y_pred[idx]}, Actual: {y_test.iloc[idx]}")
        print(f"Title: {row['title']}")
        print(f"Text: {row['text'][:300]}...")
# BERT section not changed for brevity
else:
    # Import necessary modules for BERT training
    from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    import torch

    print("Tokenizing and training BERT classifier...")

    # Convert DataFrames to HuggingFace Dataset format
    bert_train = Dataset.from_pandas(train_df[['text', 'label']])
    bert_test = Dataset.from_pandas(test_df[['text', 'label']])

    # Load BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Define tokenization function
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=512)

    # Tokenize datasets
    bert_train = bert_train.map(tokenize, batched=True)
    bert_test = bert_test.map(tokenize, batched=True)

    # Convert datasets to PyTorch format
    bert_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    bert_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Load BERT model for binary classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Set training parameters
    training_args = TrainingArguments(
        output_dir="./bert_model",              
        evaluation_strategy="epoch",           
        per_device_train_batch_size=8,           
        per_device_eval_batch_size=8,             
        num_train_epochs=2,                       
        logging_dir="./logs",                    
        save_strategy="epoch"                     
    )

    # Set up the Trainer with model and datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=bert_train,
        eval_dataset=bert_test
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the test set
    print("Evaluating BERT model...")
    predictions = trainer.predict(bert_test)
    y_pred_bert = np.argmax(predictions.predictions, axis=1)  # Convert logits to predicted labels
    print("Accuracy:", accuracy_score(test_df['label'], y_pred_bert))
    print("Classification Report:\n", classification_report(test_df['label'], y_pred_bert))

    # Display confusion matrix
    cm = confusion_matrix(test_df['label'], y_pred_bert)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Hyperpartisan", "Hyperpartisan"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - BERT Classifier")
    plt.savefig("Figure_BERT_Confusion.png")
    plt.show()

    # Save the trained BERT model and tokenizer
    model.save_pretrained("bert_model")
    tokenizer.save_pretrained("bert_model")

