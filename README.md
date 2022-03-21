

# Gender classification

Final project for the Machine Learning course

## Summary

In this project I will implement a gender classification algorithm using **Deep Neural Network**. This classification uses face of a person from a given image to tell the gender (male/female) of the given person. The result of this model is 0 or 1, representing for either female or male.
<p style="text-align:center;"><img src="https://firebasestorage.googleapis.com/v0/b/pipai212.appspot.com/o/Capture.PNG?alt=media&token=29235fe2-0b4f-4254-bbe8-e03ded794e76" width="500"></p>

## Background

I do this project for Machine Learning course in my university. This is my first ML project so it is simple and just focus on Deep Neural Network without any regularization method. I decide to implement from scratch by Python to improve my coding skill and also improve.

The problem is describe as:
*  Get an input image.
*  Output the gender prediction: male or female.



## How is it used?

First, you need to download the dataset folder.  
*  *newMale_validation* and *newFemale_validation* is used for validation.
*  Other dataset folder is used for trainning set

Second, install all the package in *requirements.txt*.  
Finally, run file *main.py*



## Data sources and AI methods
The dataset I use in this assignment is found on Kaggle.
[Gender classification.](https://www.kaggle.com/cashutosh/gender-classification-dataset)
This dataset contain about 28500 images of each class (male and female). I just use 149 first
images of each class for training and 60 last images for testing.  

The table below show the performance of model on trainning set and test set.

| Dataset     | Accuracy |
| ----------- | ----------- |
| Trainning set      | 0.99       |
| Test set   | 0.93        |

This is result when I test with a sample image.
<p style="text-align:center;"><img src="https://firebasestorage.googleapis.com/v0/b/pipai212.appspot.com/o/Test.PNG?alt=media&token=2245d600-c226-45f1-a774-6558f20ca0cd" width="500"></p>

## Challenges

*  The dataset is small and does not come from Vietnam.
*  There is not any regularization method

## What next?

*  Find another data source which include Vietnamese's face.
*  Add more regularization to improve accuracy on test set.


## References

[1] Vũ Hữu Tiệp (2018), Machine Learning cơ bản.  
[2] Neural Networks and Deep Learning (Course on Coursera)-Andrew Ng.  
[3] B.A.Golomb, D.T.Lawrence, T.J.Sejnowski (1991), SEXNET: A Neural Network Identifies Sex from Human Faces.

