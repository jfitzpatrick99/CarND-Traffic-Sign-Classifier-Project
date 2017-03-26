#**Traffic Sign Recognition** 

---

** Build a Traffic Sign Recognition Project **

This is my writeup for building a traffic sign recognition project.

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[web_image_1]: ./web-traffic-signs/sign_1.jpg "Web Traffic Sign 1"
[web_image_2]: ./web-traffic-signs/sign_2.jpg "Web Traffic Sign 2"
[web_image_3]: ./web-traffic-signs/sign_3.jpg "Web Traffic Sign 3"
[web_image_4]: ./web-traffic-signs/sign_4.jpg "Web Traffic Sign 4"
[web_image_5]: ./web-traffic-signs/sign_5.jpg "Web Traffic Sign 5"
[web_image_1_resized]: ./web-traffic-signs-processed/sign_1_resized.jpg "Web Traffic Sign 1"
[web_image_2_resized]: ./web-traffic-signs-processed/sign_2_resized.jpg "Web Traffic Sign 2"
[web_image_3_resized]: ./web-traffic-signs-processed/sign_3_resized.jpg "Web Traffic Sign 3"
[web_image_4_resized]: ./web-traffic-signs-processed/sign_4_resized.jpg "Web Traffic Sign 4"
[web_image_5_resized]: ./web-traffic-signs-processed/sign_5_resized.jpg "Web Traffic Sign 5"

[training_set_image_1]: ./examples/training_image_1.jpg "Training Set Image 1"
[training_set_image_2]: ./examples/training_image_2.jpg "Training Set Image 2"
[training_set_image_3]: ./examples/training_image_3.jpg "Training Set Image 3"
[training_set_image_4]: ./examples/training_image_4.jpg "Training Set Image 4"
[training_set_image_5]: ./examples/training_image_5.jpg "Training Set Image 5"
[training_set_image_6]: ./examples/training_image_6.jpg "Training Set Image 6"
[training_set_image_7]: ./examples/training_image_7.jpg "Training Set Image 7"
[training_set_image_8]: ./examples/training_image_8.jpg "Training Set Image 8"
[training_set_image_9]: ./examples/training_image_9.jpg "Training Set Image 9"
[training_set_image_10]: ./examples/training_image_10.jpg "Training Set Image 10"

[image_before_grayscale]: ./examples/image_before_grayscale.jpg = "Image Before Grayscale"
[image_after_grayscale]: ./examples/image_after_grayscale.jpg = "Image After Grayscale"

[image_before_random_shift]: ./examples/image_before_random_shift.jpg = "Image Before Random Shift"
[image_after_random_shift]: ./examples/image_after_random_shift.jpg = "Image After Random Shift"

[image_before_random_resize]: ./examples/image_before_random_resize.jpg = "Image Before Random Resize"
[image_after_random_resize]: ./examples/image_after_random_resize.jpg = "Image Before Random Resize"

[image_before_random_rotate]: ./examples/image_before_random_rotate.jpg = "Image Before Random Rotate"
[image_after_random_rotate]: ./examples/image_after_random_rotate.jpg = "Image Before Random Rotate"

## Rubric Points

The sections that follow address each of the
[rubric points](https://review.udacity.com/#!/rubrics/481/view) individually
and describe how I addressed each point in my implementation.  

### Dataset Summary

The dataset used for this project consists of the following:

* 34,799 training examples (training set)
* 4,410 validation examples (validation set)
* 12,630 testing examples (test set)
* 5 randomly selected examples downloaded from the web (extra test set)
* The number of unique classes/labels is 43
* The shape of each image is 32x32x3 (in RGB)

In addition, each image was cropped so that the traffic sign is the most
prominent feature in the image.

The code for setting up each dataset is located in the 1st code cell of the
IPython notebook. The code for displaying the summary data is located in the
2nd code cell of the IPython notebook.

I also augmented the training set provided with the project by developing a
jittered dataset consisting of additional images to use for the purposes of
training. This is described in detail in the "Model Training" section.

Note that assignment to training, validation and test sets was already done for
us.

### Exploratory Visualization

The exploratory visualization for my project consisted of simply choosing 10
random images from the training set and displaying each one. This approach was
chosen so that the a variety of images in the training set could be viewed
each time the cell was executed. This helped to get a decent understanding
of the data to be classified.

Here are some example images from the training set:

* ![Training set image 1][training_set_image_1]
* ![Training set image 2][training_set_image_2]
* ![Training set image 3][training_set_image_3]
* ![Training set image 4][training_set_image_4]
* ![Training set image 5][training_set_image_5]
* ![Training set image 6][training_set_image_6]
* ![Training set image 7][training_set_image_7]
* ![Training set image 8][training_set_image_8]
* ![Training set image 9][training_set_image_9]
* ![Training set image 10][training_set_image_10]

The 3rd code cell in the IPython notebook contains the code for displaying the
images.

### Pre-processing

For the purposes of doing classification, each image is converted to grayscale
prior to being given to my network. Through experimentation I noticed that a
grayscale conversion alone resulted in a 2% improvement in the validation
accuracy. I cannot say for sure why this would be the case and in some ways I
found this to be counter-inuitive since information was being thrown away. I
speculate that the reason a grayscale conversion was effective was because
this better exposed changes in gradients compared to the color images and
these gradient changes were more important than the color information, at least
for the purposes of doing traffic sign classification.

Here is an example image prior to grayscale conversion:

![Image before conversion to grayscale][image_before_grayscale]

Here is the same image after being converted to grayscale:

![Image after conversion to grayscale][image_after_grayscale]

The code for the grayscale conversion function is located in the 4th code cell
of the IPython notebook. The code for pre-processing images for training is
located in the 8th code cell of the IPython notebook. 

In addition to grayscale conversion, I also considered altering the contrast.
As described in the "Solution Approach" section, this did not result in a
noticeable improvement so I did not use this.

### Model Architecture

I decided to use LetNet for my neural network architecture. I chose
LeNet because:

1. An example implementation was readily available.
2. LeNet is known to do well on image recognition tasks.
3. LeNet is simple enough that training is possible without resorting to using
   a GPU.
4. LeNet has a reasonably small memory footprint.

The model architecture is defined in the 9th code cell of the IPython notebook.

### Model Training

To train my model I used the Adam Optimizer. After some experimentation, I
found that the following hyperparameters seemed to give the best performance
for the time I had available for training while consistently acheiving the
required validation accuracy:

| Parameter | Value |
| --------- | ----- |
| Learning Rate | 0.001 |
| Epochs | 20 |
| Batch Size | 256 |

For the purposes of training the network, I also decided to create 3 additional
"jittered" training examples for each example in the training set. This was
an idea I took from a paper written by Pierre Sermanet and Yann LeCun and
linked to in the IPython notebook. For convenience, the link to the paper is:

[http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

I used the following "jitter" functions:

* random shift - the image was randomly shifted +/- 2 pixels in the x and y direction
* random resize - the image was randomly scaled up or down by a factor in the range 0.9 to 1.1 inclusive
* random rotate - the image was randomly rotated +/- 15 degrees

Each jitter operation above required that parts of the image needed to be
filled in with replacement pixel values. For example, shifting an image
up and to the right requires replacement pixels on the left and on the bottom.
For simplicity in each case these pixels were set to zero.

In my experiments augementing the training set with jittered images seemed to
improve the consistency of my validation accuracy such that I was able to
consistently get an accuracy above 93% on each training run.

To illustrate, here are some example images before and after applying each of
the jitter functions described above:

*Images Before and After Being Shifted*

* ![Before random shift][image_before_random_shift]
* ![After random shift][image_after_random_shift]

*Images Before and After Being Resized*

* ![Before random resize][image_before_random_resize]
* ![After random resize][image_after_random_resize]

*Images Before and After Being Rotated*

* ![Before random rotate][image_before_random_rotate]
* ![After random rotate][image_after_random_rotate]

The code for augmenting the training set with the jittered training examples is
located in the 8th code cell of the IPython notebook with the jitter functions
defined in the 6th code cell. The code for training the model is located in the
12th code cell of the IPython notebook. The function that evaluates the model
against a dataset (used for displaying the validation accuracy during training)
is in the 11th code cell.

### Solution Approach

After deciding on the architecture I took an iterative approach to finding an
acceptable solution to the problem. This consisted of following steps:

1. Establish a baseline by using LeNet on the original training set with no
   pre-processing and using default values for hyperparameters. My default
   hyperparamter values were those provided in the IPython notebook without
   any changes. For reference these are learning rate: 0.001, epochs: 10
   and batch size: 128.

2. Pre-process the images to find out what pre-processing steps gave the best
   performance boost. I tried the following pre-procesing functions: grayscale
   converstion, contrast brighten, contrast darken. Contrast brighten involved
   making brighter pixels even brighter while contrast darken involved making
   darker pixels even darker. I discovered that conversion to grayscale seemed
   to give the best performance boost (about 2% improvment in validation
   accuracy) while contrast changes did not make any significant improvement.

3. Use different activation functions to see which one gave the best
   performance. From this step I determined that softsign gave the best
   performance. Sample validation accuracies for each activation function are
   provided below.

4. Augment training set with jittered images to see if this would further
   improve accuracy. To see if this was viable I started out by implementing
   the random shift jitter function. I didn't notice much of an improvment
   in validation accuracy but it did seem to make my results more consistent
   from one training run to the next. After this I implemented the other jitter
   functions random resize and random rotate. This actually caused a slight
   decrease in validation accuracy, 0.933 to 0.928 resulting in underfitting.
   I suspected this might be due to having the batch size too small or not
   enough epochs. For this reason I ignored the drop in validation accuracy and
   decided to experiment with the hyperparameters. See the next step.

5. In step 4 I saw that adding jittered examples to the training set seemed to
   reduce the validation accuracy. In this step I decided to increase the
   batch size to 256. This actually reduced the validation accuracy a bit
   further but I noticed that the validation accuracy was still trending
   downwards on the 10th epoch. Because of this I decided to double the number
   of epochs to 20. I then observed that as the training process got closer
   to the 20th epoch the validation accuracy started to stabilize at around a
   validation accuracy of between 0.93 and 0.95.

As mentioned in step 2, the softsign activation function gave the best
performance. I arrived at this decision purely by experimentation. The table
below shows typical validation accuracies for various activation functions in
order of best to worst validation accuracy.

| Activation Function | Validation Accuracy |
| ------------------- | ------------------- |
| Softsign | 0.940 |
| Tanh | 0.911 |
| Sigmoid | 0.909 |
| ReLu | 0.900 |
| Softplus | 0.879 |
| Elu | 0.850 |

*Note: Each of the results in the table above were obtained by using a learning
rate of 0.001, 10 epochs and a batch size of 128 with grayscale pre-processing.*

As can be seen from the table above, Softsign performed so well that on my
initial attempt at using it I acheived a validation accuracy of 94% without
jittering the training dataset but still doing grayscale pre-processing. I
was a little bit suspicious about this result and sure enough on subsequent
training runs, while I still got pretty good performance, validation accuracy
of above 92%, it seemed to fluctuate quite a bit from one training run to the
next.

The code for calculating the accuracy of my solution on the validation set is
in the 11th code cell of the IPython notebook. The code for calculating the
accuracy of my solution on the test set is in the 14th code cell of the IPython
notebook.

My final model results were:

* Validation set accuracy of 0.954
* Test set accuracy of 0.942

### Testing the Model on New Images

Here are five German traffic signs that I found on the web after being cropped
and resized:

1. ![Speed limit (80 km/h)][web_image_1_resized]
2. ![No passing][web_image_2_resized]
3. ![No entry][web_image_3_resized]
4. ![Yield][web_image_4_resized]
5. ![Priority road][web_image_5_resized]

The original images are as follows:

1. ![Speed limit (80 km/h)][web_image_1]
2. ![No passing][web_image_2]
3. ![No entry][web_image_3]
4. ![Yield][web_image_4]
5. ![Priority road][web_image_5]

The lighting conditions and visibility for each of the signs is very good so
I did not expect any issues classifying each of the images.

Here are the results of the prediction of my trained classifier on each image:

| Image | Prediction |
| ----- | ---------- |
| 1. Speed limit (80 km/h) | Keep Left |
| 2. No passing | No passing |
| 3. No entry | No entry |
| 4. Yield | Yield |
| 5. Priority road | Priority road |

As you can see the accuracy on these new images is 80% which matches fairly
well with what I got on the test set. In particular, I would expect that adding
more images to this extra test set would give me very similar accuracy.
Obviously because this test set is so small, one wrong answer has a significant
impact on the accruacy measurement.

The code for making the predictions on these images is located in the 18th code
cell of the IPython notebook.

Below are tables containing each of the top 5 softmax probabilities for each
image. In looking at the results below, my classifier is very confident on the
images it got right and confused regarding the image it got wrong. This is
indicated by the relatively narrow spread of probabilities for the incorrectly
classified image compared to the images that were correctly classified.

*Image 1: Speed limit (80 km/h)*

| Probability | Prediction |
| ----------- | ---------- |
| 0.426929 | Keep left |
| 0.277613 | Speed limit (30km/h) |
| 0.141507 | Speed limit (50km/h) |
| 0.0580122 | Roundabout mandatory |
| 0.0360092 | Children crossing |

As can be seen from the table above, the actual sign classification is not in
the top 5 predictions for the image. This indicates that my model potentially
has poor recall for 80 km/h speed limit signs. More investigation would be
needed to figure out why this is the case. Specifcally I would check to see
that 80 km/h speed limit signs are reasonably well represented in both the
training and validation sets and look at adding more examples of these signs
to improve this result.

*Image 2: No passing*

| Probability | Prediction |
| ----------- | ---------- |
| 0.999585 | No passing |
| 0.000368811 | Priority road |
| 2.25628e-05 | Ahead only |
| 1.01326e-05 | Vehicles over 3.5 metric tons prohibited |
| 3.66784e-06 | End of no passing |

*Image 3: No entry*

| Probability | Prediction |
| ----------- | ---------- |
| 0.999121 | No entry |
| 0.000372152 | No passing |
| 0.000263223 | Stop |
| 0.000205887 | Turn left ahead |
| 1.853e-05 | Turn right ahead |

*Image 4: Yield*

| Probability | Prediction |
| ----------- | ---------- |
| 0.982527 | Yield |
| 0.010628 | Double curve |
| 0.00661395 | Road work |
| 0.000160749 | Priority road |
| 3.0271e-05 | Dangerous curve to the left |

*Image 5: Priority road*

| Probability | Prediction |
| ----------- | ---------- |
| 0.999992 | Priority road |
| 4.07186e-06 | No passing |
| 2.42799e-06 | No vehicles |
| 1.11543e-06 | Roundabout mandatory |
| 4.14688e-07 | End of all speed and passing limits |
