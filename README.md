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
- [X] Feature Selection(Word length, Number of vowels, Contains accent, Contains unique letter combinations, Frequency of each vowel)
- [X] Metric selection
- [X] Model training
- [X] Determine training and test sets
- [X] Combining K-Fold with Grid Search


## P2: Semantic segmentation with neural networks


The problem posed is segmentation of medical images, in which the cells that appear from the background of the image must be separated.
The original data set consists of 50 pairs of images, where each pair is in a different folder and all are contained in the main directory 'Mabimi'. Image pairs consist of the original image and its mask.
The objective is to obtain neural network models that can satisfactorily segment medical images only from the original image, to later evaluate them. One of the models will be based on an existing network (we will choose UNet), and the others will be modifications of it (this way we can make comparisons more easily).


<img width="849" alt="Captura de pantalla 2023-11-22 a las 8 24 02" src="https://github.com/maribel95/Machine-learning/assets/61268027/25088701-6287-43f6-8b3d-3f6b5bb52c30">


### Consideres solutions:

In every neural network problem there are a series of aspects to address before moving on to the implementation of the solution. In this case, three different neuronal models have been proposed. The idea is to observe the operation of each one and analyze the performance of each network for the same problem. The idea is to use a neural network as a basis, apply some changes to its architecture and see the impact of these changes.








