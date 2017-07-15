# Emotion-Recognition-using-Facial-Landmarks
This project is made for recognizing Human facial expressions without using convolutional neural nets. It uses non feature learning approaches such as Facial Landmarks.The best accuracy for Fer2013 (as I know) is 67%, the author trained a Convolutional Neural Network during several hours in a powerful GPU to obtain this results.Here is a much simpler (and faster) approach by extracting Face Landmarks and feeding them to a multi-class SVM/ Logistic Regression/ Randomforest  classifier.

## Requirements
You will need the following to run the above:
 * Python3.5
 * Numpy
 * OpenCV
 * SKLearn
 * Dlib
 
## How to run the code
1. *  Download the 'fer2013' dataset from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
   * run the following script to convert the .csv file into images
```
$ python3 csv_to_images.py -f <path to the fer2013.csv file> -o <output folder to save images in>
```
2. Download the landmarks file from [here](Download the the 'fer2013' dataset from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

3. train the model by running the following in the terminal
```
$ python3 train.py
```

## Results






