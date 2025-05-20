# 🧠 NLP Pipeline: Text Classification, Summarization, and Clustering

This project explores three fundamental tasks in Natural Language Processing (NLP): **Text Classification**, **Text Summarization**, and **Text Clustering**. Each component is implemented using state-of-the-art models and libraries such as Hugging Face Transformers and Scikit-learn.

---

## 📁 Project Structure

project/
├── classification TRUE.ipynb # Text Classification using Transformers
├── NLP(summrazition).ipynb # Summarization using BART
├── Clustring.ipynb # Clustering for Topic Discovery
├── requirements.txt # Project dependencies

---

## 🔹 1. Text Classification

**File:** `classification TRUE.ipynb`

- Uses a pre-trained BERT-like transformer from Hugging Face.
- Loads and preprocesses product reviews with associated ratings.
- Trains and evaluates a classification model for sentiment or category prediction.

**🔧 Input:**  
Customer reviews + ratings  
**🎯 Output:**  
Predicted sentiment or product category  

**✅ Features:**  
- Data cleaning and preprocessing  
- Tokenization using `AutoTokenizer`  
- Model training with evaluation metrics (Accuracy, Precision, Recall, F1)  

---

## 🔹 2. Text Summarization

**File:** `NLP(summrazition).ipynb`

- Applies extractive and abstractive summarization on review data.
- Filters reviews by product category and summarizes using the `facebook/bart-large-cnn` model.

**🔧 Input:**  
Raw review text  
**🎯 Output:**  
Concise summary of reviews  

**✅ Features:**  
- Summarization using Hugging Face pipelines  
- Easy filtering and customization by product/category  

---

## 🔹 3. Clustering & Topic Discovery

**File:** `Clustring.ipynb`

- Applies vectorization (TF-IDF or embeddings) to convert text into numerical features.
- Performs clustering using **KMeans** or **Hierarchical Clustering**.
- Visualizes results and evaluates clustering using Silhouette Score.

**🔧 Input:**  
Unlabeled textual data  
**🎯 Output:**  
Groups of similar topics or clusters  

**✅ Features:**  
- Dimensionality reduction with PCA (optional)  
- Visual representation of clusters  
- Insights into hidden topics in datasets  

---

## 🛠️ Requirements

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
Required Libraries:
transformers
scikit-learn
pandas
matplotlib
seaborn
scipy
gradio
jupyter
