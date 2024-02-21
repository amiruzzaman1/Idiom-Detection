# Idiom-Detection
This project pioneers advancements in Natural Language Processing (NLP) by introducing a deep neural network approach to address challenges in interpreting multi-word sentences, detecting idioms, and classifying text. Focusing on text classification, it evaluates various models, with BERT outperforming others with an F1 score of 0.7980 on the Dev Dataset. Noteworthy results include Linear SVM with Balanced Class achieving a high F1 score of 0.8457 on the Few-Shot Dataset and Multinomial Naive Bayes excelling with a score of 0.5284 on the Zero-Shot Dataset. These findings underscore the importance of diverse datasets for robust model training and contribute to the ongoing evolution of NLP technology.

## Dataset Links

- [Dev Dataset](https://raw.githubusercontent.com/H-TayyarMadabushi/AStitchInLanguageModels/main/Dataset/Task1/SubTaskA/EN/ContextExcluded_IdiomExcluded/dev.csv)
- [Test Dataset](https://raw.githubusercontent.com/H-TayyarMadabushi/AStitchInLanguageModels/main/Dataset/Task1/SubTaskA/EN/ContextExcluded_IdiomExcluded/test.csv)
- [Train Few-Shot Dataset](https://raw.githubusercontent.com/H-TayyarMadabushi/AStitchInLanguageModels/main/Dataset/Task1/SubTaskA/EN/ContextExcluded_IdiomExcluded/train_few_shot.csv)
- [Train One-Shot Dataset](https://raw.githubusercontent.com/H-TayyarMadabushi/AStitchInLanguageModels/main/Dataset/Task1/SubTaskA/EN/ContextExcluded_IdiomExcluded/train_one_shot.csv)
- [Train Zero-Shot Dataset](https://raw.githubusercontent.com/H-TayyarMadabushi/AStitchInLanguageModels/main/Dataset/Task1/SubTaskA/EN/ContextExcluded_IdiomExcluded/train_zero_shot.csv)


## Colab Notebook (Click to View)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L5YNwjAW66fxhDUFK45fH796QUF9jyxo?usp=sharing)



## EDA (Exploratory Data Analysis)
![Visualizations of Labels for different datasets](https://github.com/amiruzzaman1/Idiom-Detection/assets/68743925/b23fb076-d1f3-47ab-b9e2-f030ce8da87f)

*Fig1 : Visualizations of Labels for different datasets*


![image](https://github.com/amiruzzaman1/Idiom-Detection/assets/68743925/6c7f021a-db71-4d16-83f4-5d117ac3778e)
![image](https://github.com/amiruzzaman1/Idiom-Detection/assets/68743925/2c8b3dd8-6e14-4e50-99aa-f164b38fe625)
![image](https://github.com/amiruzzaman1/Idiom-Detection/assets/68743925/962c4cd9-5d06-4a7e-a058-6bf66e2e24eb)
![image](https://github.com/amiruzzaman1/Idiom-Detection/assets/68743925/a631bccb-8e85-444c-8131-510ca7007805)
![image](https://github.com/amiruzzaman1/Idiom-Detection/assets/68743925/9ca44847-ff0a-4a6d-8ef6-73f0d5ceeca5)

*Fig2 : Histograms of sentence lengths*


# Results
## F1 Scores of Different Models:
| Classifier                                    | Features/Approach                | Few-Shot | Zero-Shot | One-Shot | Dev   |
| --------------------------------------------- | --------------------------------- | -------- | --------- | -------- | ----- |
| Linear SVM with Unbalanced Class              | Uni-gram                         | 0.7441   | 0.5222    | 0.4965   | 0.5779|
| Linear SVM with Balanced Class                | Uni-gram                         | 0.8457   | 0.5216    | 0.6282   | 0.5482|
| SVM (Hyper Parameters)                        | Uni-gram                         | 0.8384   | 0.5399    | 0.6119   | 0.5407|
| Linear SVM with Balanced Class Weights and Bigrams | Bigram                       | 0.8550   | 0.3505    | 0.5249   | 0.4271|
| Logistic Regression                            | Uni-gram                         | 0.8233   | 0.5348    | 0.6148   | 0.5872|
| Logistic Regression (Using Stemmer)            | Uni-gram                         | 0.8139   | 0.5343    | 0.6514   | 0.5872|
| Logistic Regression (with Bigrams)             | Bigram                          | 0.8501   | 0.4114    | 0.5249   | 0.4227|
| Multinomial Naive Bayes (Unigrams)            | Uni-gram                         | 0.4692   | 0.5235    | 0.4429   | 0.5348|
| Multinomial Naive Bayes (Bigrams)             | Bigram                          | 0.5066   | 0.3945    | 0.4088   | 0.4144|
| Multinomial Naive Bayes (Lemmatization)       | Uni-gram + Lemmatization         | 0.8043   | 0.5284    | 0.6958   | 0.5345|
| LSTM                                          | N/A                             | 0.4088   | 0.2358    | 0.4088   | 0.4088|
| Bi-directional LSTM                            | N/A                             | 0.6308   | 0.4832    | 0.5677   | 0.5621|
| XGBoost                                       | N/A                             | 0.6917   | 0.4318    | 0.4863   | 0.5168|
| BERT                                          | N/A                             | 0.7819   | 0.2358    | 0.7676   | 0.7980|
| RoBERTa                                       | N/A                             | 0.6895   | 0.6729    | 0.6385   | 0.5870|

## Accuracy of Different Models:
| Classifier                                   | Features/Approach                  | Few-Shot | Zero-Shot | One-Shot | Dev   |
| -------------------------------------------- | ---------------------------------- | -------- | --------- | -------- | ----- |
| Logistic Regression (Unigrams)               | Uni-gram                          | 0.8509   | 0.5424    | 0.7164   | 0.6439|
| Logistic Regression (Using Stemmer)          | Uni-gram                          | 0.8468   | 0.5404    | 0.6708   | 0.6439|
| Logistic Regression (Bigrams)                | Bigram                            | 0.8716   | 0.4182    | 0.6915   | 0.6667|
| Multinomial Naive Bayes (Unigrams)           | Uni-gram                          | 0.7081   | 0.5259    | 0.6998   | 0.6957|
| Multinomial Naive Bayes (Bigrams)            | Bigram                            | 0.7205   | 0.4058    | 0.6915   | 0.6894|
| Multinomial Naive Bayes (Lemmatization)      | Uni-gram + Lemmatization          | 0.8406   | 0.5383    | 0.7474   | 0.5714|
| BERT                                       | N/A                               | 0.8219   | 0.3085    | 0.8137   | 0.8302|



