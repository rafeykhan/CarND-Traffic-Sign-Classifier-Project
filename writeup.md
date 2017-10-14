# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results in this report


[//]: # (Image References)

[image1]: ./1.JPG "Visualization"
[image2]: ./3.JPG "Grayscaling"
[image3]: ./2.JPG "augmentation"
[image4]: ./4.JPG "stats"
[image5]: ./5.JPG "Traffic Signs"
[image6]: ./6.JPG "softmax1"
[image7]: ./7.JPG "softmax2"


---

### Data Set Summary & Exploration

I used the Python and NumPy library to find the amount of samples and the shape of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a histogram showing how the data in the training and validation set is distributed.

![alt text][image1]

#### Data Augmentation

It can be seen that the data is unevenly distributed. Some of the classes have very few samples which can affect how the model is trained.

For this reason, I decided to generate additional data for the classes that have fewer samples.
To add more data to the the data set, I applied the following techniques on the original images using OpenCV:
**rotation**
**translation**
**zoom**
**distortion**

Here is an example of an original image and the augmented images:

![alt text][image3]

The difference between the original data set and the augmented data set is only in the training set. For all the classes with fewer than 500 samples, two randomly picked image operations are applied to every image in that class. In short, the samples in those classes are trippled.

#### Preprocessing and Normalizing Images in the Data Set

As a first step, I decided to convert the images to grayscale because removing color information increases the speed of training and better learn and extract features. Training color images took longer without any improvement in accuracy.
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it makes it easier for the network to learn.
**(pixel - mean)/std**

### Design and Test a Model Architecture

My model is based on the LeNet Architecture. To counter overfitting I added dropout layers and applied L2 regularization. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution     	|5x5 filter, 1x1 stride, 'valid' padding, outputs 28x28x6 	|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Dropout					|	keep_prob 0.8											|
| Convolution 	    | 5x5 filter, 1x1 stride, 'valid' padding, outputs 10x10x16      									|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Dropout					|	keep_prob 0.8											|
| Fully connected		| input 400, output 120         									|
| ELU					|												|
| Dropout					|	keep_prob 0.5											|
| Fully connected		| input 120, output 84         									|
| ELU					|												|
| Dropout					|	keep_prob 0.5											|
| Softmax				| output 43 classes       									|
|						|												| 


#### training the model

In the model, I used an ELU activation instead of RELU because the model converged faster and with better accuracy. For training, I used Adam Optimizer with learning rate of **0.01** . I chose batch size of 128 and trained the model for 100 epochs, which allowed the model reach approximately **98%** validation set accuracy. 

LeNet architecture was used to classify handwritten numbers input as images. Therefore, it seemed as a good starting point for traffic sign images. However the validation accuracy was not satisfying, around 90% training on 15 epochs. Preprocessing the data by grayscaling and normalizing improved learning, boosting validation accuracy to above 93%. I then switched activation function to ELU which helped the network converge better and improve the accuracy by 2%.
To improve generalization of classifier and prevent overfitting, I adjusted the architecture by adding dropout layers with keep probabilities of 0.8 and 0.5. L2 regularization with 0.00001 was also applied into fully connected layers.
Applying these changes over 100 epochs of training while also augmenting data improved the performance immensely.

My final model results were:
* validation set accuracy of 97.5% 
* test set accuracy of 95.8%

**Incorrect probabilities by class are illustrated in the following bar chart**

![alt text][image4]



### Test a Model on New Images

Here are five German traffic signs that I found on the web:

 ![alt text][image5]
 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h)   									| 
| Speed limit (100km/h)     			| Speed limit (100km/h) 										|
| No entry  | No entry											|
| Go straight or right	      		| Go straight or right					 				|
| Yield			| Yield      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.8%. However, if one of the images were of "Speed limit (80km/h)" it might have been incorrectly classified as, according to the statistics on the model, around 2 in every 5 samples on the test set were predicted incorrectly.

#### softmax probabilities for each prediction.

For these images, the model is pretty sure that this is a certain sign (probabilities greater 0.9), and that happens to be the correct prediction.

![alt text][image6] 

![alt text][image7] 




