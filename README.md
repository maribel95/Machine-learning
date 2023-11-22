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


### Considered solutions:

In every neural network problem there are a series of aspects to address before moving on to the implementation of the solution. In this case, three different neuronal models have been proposed. The idea is to observe the operation of each one and analyze the performance of each network for the same problem. The idea is to use a neural network as a basis, apply some changes to its architecture and see the impact of these changes.

#### UNet

In the case of evaluating an existing architecture we will use a UNet. This is because UNet is specially designed for image segmentation (it has skip connections that retain image information when reconstructing it after extracting features), and because we have also taught it in class and we already have an implementation in the teacher's github.

<img width="748" alt="Captura de pantalla 2023-11-22 a las 8 30 02" src="https://github.com/maribel95/Machine-learning/assets/61268027/de6a9516-3179-4562-b56e-c63a70e90226">

And then we have built our own models, carrying out different experiments:

#### PotiNet

UNet variation that will get rid of skip connections. In this way we can evaluate how it affects having or not having this technique implemented. We expect the results to be worse, since now the decoder will receive half of the features (which will no longer arrive from the encoder), and will negatively impact the network's ability to retain information across the layers and will result in a reduction in precision and an increase in error.


#### XopiNet

Variation of the UNet in which we will add more layers of depth. You may get better results, but you run the risk of overfitting if you adjust too closely to the training data. Additionally, it will be computationally more expensive.
In the next main section (Experiments performed) we will see the structure of the models and the layers themselves.

### Data treatment:

Because we will use the Pytorch library, we will have to apply a series of changes to the images in order to work with them properly.
When analyzing the data we quickly realize that the images of the original masks have several details to take into account:
- The first is that the color space of the masks is not binary, that is, there is a margin of gray that does not interest us. We are not interested in grays, so we apply a threshold where the pixels become completely white, and the rest become black.
- The second thing is that there was a specific mask image whose color space was different from the rest (cell_14 mask), specifically RGB. We have simply converted it to grayscale and replaced it in the dataset. This means that to execute the provided code you must download the modified dataset (there is a link at the end of the document that also includes the weights to download).

Once these initial considerations have been resolved, we can now proceed to create our dataset.

We will create our own dataset class where we will store all the images along with their masks. We will apply the necessary transformation to be able to have them in a format that fits our way of working with neural networks. This transformation will consist of resize to 224x224 . Later we turn them into tensioners. Finally we will normalize the source images, dividing by the maximum pixel value.
Once this is done, we divide the data set into training and test, giving a size of 80% to the training (40 images) and 20% to the test (10 images).
Then we will perform a random split of the training data and the test data, to finally load the data in batches.
In this very particular case, as we only have 50 samples (it is a very low number) we have decided to use batches of size 1.

A large batch size for training would not have been correct since, for example, with a size of 10, we would only readjust the weights 4 times (taking into account that our training set is made up of 40 images).

### Metrics:

We have selected two metrics to evaluate the performance of the model: IoU (Intersection over Union) and Dice Score. Using two will give us a broader view of the models' performance.

#### Intersection over Union

As its name indicates, the calculation of this metric consists of dividing the intersection between the union of both sets, with A being the ground truth and B the result of the segmentation in the image. IoU measures the similarity between two regions and may not be the best choice for binary segmentation, where pixels are classified rather than identifying entire regions.

<img width="321" alt="Captura de pantalla 2023-11-22 a las 8 43 40" src="https://github.com/maribel95/Machine-learning/assets/61268027/cf98353f-5dbd-4c28-8620-a50a7dcf122c">



#### Dice Score

The Dice Score takes into account both the positives found and those that were not found, penalizing if the algorithm does not find them.

Although in the case of binary segmentation, the Dice Score is the most appropriate metric to evaluate the quality of the segmentation. This is because it is a metric that is well suited to evaluating the similarity between two collections of pixels, such as segmentation results and reference data, and in fact is commonly used in the context of binary segmentation in images. medical.

Representation with operations between sets

<img width="160" alt="Captura de pantalla 2023-11-22 a las 8 43 32" src="https://github.com/maribel95/Machine-learning/assets/61268027/c931399d-57e5-47f8-9f09-0a99416fb9d4">














