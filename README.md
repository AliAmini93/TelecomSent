# TelecomSent: Targeted Sentiment Analysis for Telecoms


## Project Overview
This project focuses on extracting detailed and actionable insights from social media discussions about three major telecom operators: MTN. The analysis is based on data from two primary social media platforms: [Twitter](https://twitter.com/) and [Facebook](https://web.facebook.com/).

We employ both traditional machine learning methods and state-of-the-art deep learning techniques, including BERT, to automatically identify and extract key descriptors from user opinions. These descriptors are then used to generate structured summaries of the sentiments expressed, which can be utilized by telecom companies to identify customer pain points and measure performance against competitors. Similarly, customers can use these summaries to make informed choices about their telecom providers.

For supervised learning tasks, we developed a custom human-annotated dataset, referred to as TelecomSent, containing 5,423 social media posts. Each post references one or more telecom providers, offering a rich dataset for sentiment analysis.

The core components extracted from these posts include the target telecom, the specific service aspect mentioned, and the sentiment expressed towards that aspect. This methodology falls under Targeted Aspect-Based Sentiment Analysis (TABSA).

1. Python 3.6+
2. TensorFlow
3. Access to a GPU (or use Google Colab)
4. Scikit-learn
5. [BERT-Base (Google's pre-trained models)](https://github.com/google-research/bert)
6. NLTK (Natural Language Toolkit)
7. NumPy 1.15.4
8. PyTorch 1.0.0

## Results Summary
The table below summarizes the results achieved using various machine learning and deep learning approaches. We evaluated the models using strict accuracy, Macro-F1 score, and AUC, with results provided for both aspect category detection and sentiment classification.

| **Model**        | **Aspect Accuracy** | **Aspect F1** | **Aspect AUC** | **Sentiment Accuracy** | **Sentiment AUC** |
|------------------|---------------------|---------------|----------------|------------------------|-------------------|
| **RF-TFIDF**     | 0.540               | 0.392         | 0.615          | **0.958**              | 0.737             |
| **RF-word2vec**  | 0.391               | 0.115         | 0.538          | 0.956                  | 0.533             |
| **LR-TFIDF**     | 0.390               | 0.414         | 0.532          | 0.877                  | 0.508             |
| **LR-word2vec**  | 0.365               | 0.229         | 0.482          | 0.918                  | 0.487             |
| **LSTM**         | 0.705               | 0.231         | -              | 0.705                  | -                 |
| **BERT**         | **0.748**           | **0.791**     | **0.963**      | 0.937                  | **0.961**         |

## Running the Models
You can run the models by accessing the respective Jupyter notebooks provided in the [Scripts]([https://github.com/davidkabiito/Sentitel/tree/master/Code](https://github.com/AliAmini93/TelecomSent/tree/main/Scripts)) directory.

- **Random Forest with TFIDF**: [Run Notebook]([https://github.com/davidkabiito/Sentitel/blob/master/Code/random_forest/tfidf/T-ABSA_random_forest_tfidf_model.ipynb](https://github.com/AliAmini93/TelecomSent/tree/main/Scripts/random_forest/tfidf))
- **Random Forest with Word2Vec**: [Run Notebook]([https://github.com/davidkabiito/Sentitel/blob/master/Code/random_forest/word2vec/T-ABSA_random_forest_word2vec_model.ipynb](https://github.com/AliAmini93/TelecomSent/tree/main/Scripts/random_forest/word2vec))
- **Logistic Regression with TFIDF**: [Run Notebook](https://github.com/davidkabiito/Sentitel/blob/master/Code/LR/tfidf/T-ABSA_LR_tfidf_model.ipynb)
- **Logistic Regression with Word2Vec**: [Run Notebook](https://github.com/davidkabiito/Sentitel/blob/master/Code/LR/word2vec/T-ABSA_LR_word2vec_model.ipynb)
- **BERT Implementation**: [Run Notebook](https://github.com/davidkabiito/Sentitel/blob/master/Code/BERT/BERT_SentiTel.ipynb)
- **LSTM Model**: [Explore Code](https://github.com/davidkabiito/Sentitel/tree/master/Code/LSTM)
