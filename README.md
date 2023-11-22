# Machine-learning

Machine learning is a subset of artificial intelligence that allows a system to learn and improve autonomously using neural networks and deep learning, without having to be explicitly programmed, through the ingestion of large amounts of data.

Two projects involving different approaches to solving machine learning problems have been carried out.

## P1: SVM model for classify words between English and Catalan

The problem to be solved is based on the creation of a machine learning model that is capable of classifying a given word according to whether it belongs to the English or Catalan language.

To do this, we are provided with a dataset with approximately 1000 words in English and 1000 in Catalan (they are the same words). It has three columns, where two of them are the word in each language, and the remaining one is the id of the row.

The algorithm to be used will be a Support Vector Machine (SVM). SVM is based on the idea of finding a separating hyperplane between two classes of data in a high-dimensional feature space. They are very effective in binary classification problems and can also handle non-linear problems using kernel techniques.

### Data analysis

The processing of the data set is a very important stage when attacking any problem. We will not be able to obtain good results if the data is also not good.

Although the initial data is what it is, we can try to obtain new features from the existing ones: select relevant features, delete features if we see that they are not relevant, eliminate missing values, correct data errors and normalize variables.

A good treatment of the data set can significantly improve the accuracy and performance of the machine learning models used later.
​
### To do:

- [X] Data set preparation
