# Vehicle-Detection

Refer to Project submission.ipynb for the pipeline

# The goals of this project are the following:
1.  Create and train a deep learning model to classify if patch image is either a car or non-car.
2.  Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
3. Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
4.  Estimate a bounding box for vehicles detected.

# Create and train deep learning model to classify car vs non-car:
To create a deep learning model, I used keras and transfering learning.  

First I defined a sequential model.  I added a Lambda layer to normalize the input data so that the mean is zero and the data is +/- 0.5.  Then I added an Xception model without top(meaning all the fully connected layer are not included).  Then I added GlobalMaxPooling2D(), and 2 dense layers.  The last dense layer will only have one neuron so that it predict if the input data is a car or not. (0 = non-car, 1 = car).

| Layer (type)  | Output Shape  |  Param # |
| ------------- | ------------- |----------|
| lambda_1 (Lambda)  | (None, 64, 64, 3)  | 0 |
| xception (Model) | multiple  |20861480|
| global_max_pooling | (None, 2048)  |0|
| dense_1 (Dense) | (None, 1024)   |2098176|
| dense_2 (Dense) | (None, 1)  |1025|

Total params: 22,960,681
Trainable params: 22,906,153
Non-trainable params: 54,528

Once the model is built.  I train the model with the dataset.
The dataset has images (64X64) of car and non-car.  The non-car may include road, trees, guardrails.etc.  
![Non-Car](https://github.com/Tak-Au/Vehicle-Detection/blob/master/extra17.png "Non-Car")
![Car](https://github.com/Tak-Au/Vehicle-Detection/blob/master/image0001.png "Car")

I used opencv to read all images and store the image as X_data.  I assign 0 for non-car and 1 for car and store in y_data.  
I also augment the images by flipping vertical and horizontal.  I also augment by resizing the image.(Cell 3)    
Once all the images and their class are stored in X_data, and y_data, I used sklearn to split the data to training set and testing set (80% and 20% respectively). 

I train the model using the training set via 2 epochs.  I also set validation to .2 so that 20% of the training set will be used as validation so that I can see if the model is overtrained.

After the training is done, I score the model using the test set. The model scored 99.7%.  

# Implement a sliding-window technique:
I implement the sliding window by selecting patches in a image from up down and left right.  Because the road is at the bottom of the image, the starting point for y I select to be 350.  The patch size started out to be 90X90.  However, as the patch y value increase (going down), the patch size increases.  The equation I used for patch size is 90+(y-350) where y is the top of the patch).  The stridex (stride in x direction) also increase as the patch size increase, the equation i used for stridex is 2+(y-350).  The stridey is set to 25.  Everytime I get a new patch, I resize the image to 64,64 to allow for the model to classify.  When the model predict each frame, it will assign a probability if the frame has a car.  However, there are times that the model is incorrect.  There might be false positive(predict car when no car exist) or false negative(predict non-car when car exist).  I solve this problem by using heat map.  The idea of a heat map is that as the sliding window goes through each frame, adjacent frame for a car will classify as a car, the heat map will stack up all the prediction so that one can see the spot where the car prediction appear alot of times.  This way, even if there are some false postive or false negative, as long as the model is good, the overall result will be accurate.  

![Original](https://github.com/Tak-Au/Vehicle-Detection/blob/master/test6.png "Original")

![sliding window](https://github.com/Tak-Au/Vehicle-Detection/blob/master/download.png "Sliding window")

# Implement Heat Map:
To creat a heat map, I create an matrix of zeros that has the same size as the image.  Then I add each cells within a patch of the heat map with the model probablity value of the image frame.  The value will stack up as the sliding window goes through the heat map.  

![Heat Map](https://github.com/Tak-Au/Vehicle-Detection/blob/master/Heatmap.png "Heat Map")

Oncee the heat map is built, I use a threshold value to reset all cell value < threshold to zero.  The non zero values will show where the car was detected.  Then I used the labeled_bboxes function to find all boxes where the value is non zeros.  The boxes will be the location where the model believe the car exist.  However, I find that even with heat map, there can still be false postive.  The reason is because the threshold value can't set too high or too low.  If the value is set too high, it causes more false negative and too low will cause false postive.  

To combat false positive, I use the boxes that the heat map generated to extract the patches within the image and use the model to perform prediction again.  This time I set the threshold of the prediction so that if the result comes out to be > 95% confident, then it will classify the fame to be truly a car.  Then I draw the frame with the prob value on top of the frame to the Image. 

![Result](https://github.com/Tak-Au/Vehicle-Detection/blob/master/imagewithprob.png "Result")

# Result video
This is the result video when using the pipeline.  
https://github.com/Tak-Au/Vehicle-Detection/blob/master/test_videoresultdl.mp4

https://github.com/Tak-Au/Vehicle-Detection/blob/master/project_videoresultdl.mp4

# Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?
The pipeline fails when the car doesn't match the patch frame size.  In order for the pipeline to process the video without too much time, the stride can't be too small.  There's a balance between the speed and accuracy.    

One way to improve the pipeline to use other technique to detect region of interest ROI.  The (2) state of the art model is Faster RCNN and YOLO.  Both technique don't use sliding window, instead, the model will create region proposals where the object may be at.  The speed is also faster than using sliding window.  
