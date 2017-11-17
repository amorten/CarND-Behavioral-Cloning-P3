# **Behavioral Cloning** 

## Writeup 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
  
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images  
* Train and validate the model with a training and validation set  
* Test that the model successfully drives around track one without leaving the road  
* Summarize the results with a written report


[//]: # (Image References)

[center_center]: ./writeup_images/center_center.jpg "Center driving, central camera"
[center_right]: ./writeup_images/center_right.jpg "Center driving, right camera"
[recover_center]: ./writeup_images/recover_center.jpg "Recovery driving, central camera"
[recover_right]: ./writeup_images/recover_right.jpg "Recovery driving, right camera"
[unflipped]: ./writeup_images/unflipped.jpg "Image before flipping"
[flipped]: ./writeup_images/flipped.jpg "Image after flipping"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:  

* [model.ipynb](./model.ipynb) - contains the script to create and train the model (my contribution)
* [drive.py](./drive.py) -  a script for driving the car in autonomous mode (created by Udacity)
* [model.h5](./model.h5) - contains a trained convolution neural network, created by [model.ipynb](./model.ipynb)
* [writeup_report.md](./writeup_report.md) - describes the the project in detail 
* [video.mp4](./video.mp4) - the video output showing the car driving autonomously around the track


#### 2. Submission includes functional code
Using the Udacity provided simulator and my [drive.py](./drive.py) file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.ipynb](./model.ipynb) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The network architecture is defined in cell #8 of [model.ipynb](./model.ipynb).
The model consists of a convolutional neural network based on the network architecture used by the autonomous 
 vehicle team at nVidia. The architecture is [described on their website](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).
 
 My version is somewhat different: I use the RGB color space, and I add a dropout layer to reduce overfitting. I also use ReLU activation functions after the fully connected layers, which is different from how the network was presented in class video (in the class video, there were no activation functions between fully connected layers).

The network starts off by normalizing each of the RGB color channels. Then, the image is cropped to remove the region above the road and the car at the bottom. Three two-dimensional convolutional layers are then applied with 5x5 filter sizes, a stride of 2 along each axis, and depths increasing from 24 to 48. Another two convolutional layers are  then applied, this time with 3x3 filters, a stride of 1 along each axis, and depths of 64. ReLU activation functions are applied after each convolution.

The network is then flattened and dropout is applied (with 30% of connections dropped). Finally, four fully connected layers are applied with decreasing output sizes of 100, 50, 10, and 1. ReLU activation functions are applied after each fully connected layer, except for the final one.


#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (in cell #8 of [model.ipynb](./model.ipynb)). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The train/validation/test split is done in cell #6 of [model.ipynb](./model.ipynb). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually (cell #8 of [model.ipynb](./model.ipynb)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

The final cell (cell #11) of [model.ipynb](./model.ipynb) shows a plot of steering angle for the full training data (including validation and test) in chronological order before shuffling of the data. The plot shows that approximately the first half has normal driving steering angles, while the second half of the data set has very large steering angles used to recover from the side of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My strategy for designing a model architecture was to first use a well know architecture and then modify it if needed. So, I used nVidia's architecture as described earlier in this writeup. I thought this model might be appropriate because it was specifically used by nVidia to predict steering angles from road images.

In order to gauge how well the model was working, I shuffled the data and then split my image and steering angle data into a training and validation set (80%-20% split), but only after setting aside 20% for final testing. 

The first time I ran the model, the mean squared error of the validation set would oscillate every other epoch. So, I thought more carefully about the network architecture and decided I should put nonlinear activation functions between the fully connected layers, unlike how it was presented in class. After that, the validation loss was smoother and smaller.

The value of the mean squared error at approximately 0.03 was much smaller than I expected, so I thought the model was probably overfitting, even though the validation loss did not rise above the training loss. To reduce the risk of overfitting, I added a Dropout layer, but the training and validation loss both remained smaller than expected. I think what is happening is that the training and validation data are not actually sufficiently independent. When I shuffle the full data set at the beginning it cuases training and validation images to be neighboring frames in the video. Neighboring frames in the video likely have similar images and similar steering angles. To reduce overfitting, I think the training and validation (and test) datasets should be taken from completely indpendent drives around the track. I have not you implemented this, however, since the car is able to successfully drive around track, even though there is potentially for some overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track at first, so I collected more driving data in the region of the track where to trouble occured. I only had to do this once!

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture

The final model architecture (cell #8 of [model.ipynb](./model.ipynb)) is fully described earlier in this write-up!

#### 3. Creation of the Training Set & Training Process

Example training images are plotted in cell #5 of [model.ipynb](./model.ipynb).

To capture good driving behavior, I first recorded one lap around  track #1 using center lane driving. Here is an example image of center lane driving (captured by the central camera):

![center lane driving, center camera][center_center]

Images from the right and left cameras were also used, with a steering angle adjustment of plus or minus 0.3. Here's an image of center lane driving captured by the right camera:

![center lane driving, right camera][center_right]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn what to do if it strayed to far to the left or right. These images show what a recovery looks like starting from the side of the road captured by different cameras:

![recovery driving, central camera][recover_center]
![recovery driving, right camera][recover_right]

To augment the data sat, I also flipped images and angles thinking that this would remove any bias in steering to the left or right. For example, here is an image that has then been flipped:

![camera image][unflipped]
![flipped camera image][flipped]

At first the model did not perform well. So, I inspected time series of the recovery driving and discovered that I accidentally sometimes used the keyboard as steering input instead of the mouse. Whenever I used the keyboard, steering angles would oscillate between large and small values, which was wrong. I had to delete those regions from the training set and redo them by driving some more, making sure to use the mouse as input.

After the collection process, I had approximately 40000 image samples (including flipped images and views from all three cameras). All additional processing (normalizing and cropping) was handled by the neural network.

I randomly shuffled the data set and put 20% of the data into a test set. Of the remaining data, I put 20% into a validation set. The remaining data was used for training. 

I used this training data for training the model. The validation set helped determine whether the model was over- or under-fitting. However, as described previously in this write-up, I believe the model is still probably overfitting, because the training and validation data sets are too similar. One piece of evidence for overfitting is that the actual value of the mean square error is much smaller than one would expect from the variance in the steering angle for similar recovery situations. Another piece of evidence for overfitting is that the validation loss continues to decrease through 13 epochs, even the the driving appears to get worse after about 7 epochs.

The ideal number of epochs was 7 as evidenced by the car driving fairly well around the track. A standard way to choose the number of epochs would have been to select an epoch where the validation error stops decreasing and does not rise above the training error. However, in this case, I think that the validation and training sets may not have been sufficiently independent, so further decreases in the validation error after epoch 7 (or so) may be coinciding with overfitting to the training data. Of course, the car still manages to correctly drive around the track, so the extent of the overfitting, if any, is not catastrophic.

I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 3. Simulation

I used Udacity's version of [drive.py](./drive.py) to produce a [video](video.mp4) of the car driving one time around the track. The car always remains on the drivable part of the road. Yay!