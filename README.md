# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This is a short walkthrough of the training approach used to solve the problem of Udacity's third project "Behavioral Cloning".

The logic is based in model.py while drive.py handles the driving.

---

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Data Collection
--
First I've tested the model on the udacity data only but found that the behavioural of the car was still wobbly and unsafe.
I tried collecting more data with smoother steering and a lot of straight driving examples. 

I collected +35k images with better performance but the result wasn't good, still don't know why so I turned back to my previous dataset collected. 


## Preprocessing
--
I'm also resizing the image to (60, 25) and normalize it. I'm using left, right and center images randomly with a steering shift value and I'm also flipping the images .


I'm splitting my dataset into train/validation set with a factor of 0.8 (which means 80% is training, 20% is validation/test)


## Network / Training
--
The data which is loaded directly from my drive is feed into a Keras fit_generator. My generator picks a random batch of my dataset, picks one of the images (left, center, right) and might flip the image. Then the generator yields the batch and feeds it to the network. This continues until the whole dataset is used.

![Network] (https://github.com/waelHamed/Behavioral_Cloning/blob/master/model-visualization.png)


The network consists of five convolutional layers, followed by three fully connected layers. I have added Dropout Layers and SpatialDropout Layers to prevent overfitting.


## Testing / Autonomous Driving

Since the model always sets its weights a little bit different, the behaviour of the car wasn't 100 % predictable. but I've a big picture now and need to work now on adding a recovery images so I can recover from disaster even if the model was working good, but there are some points if the model wasn't trained enough it will not move as I wish. 
