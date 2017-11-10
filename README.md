# Vehicle-Detection

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
![Non-Car](https://github.com/Tak-Au/Vehicle-Detection/blob/master/extra17.png, width=200))
![Car](https://github.com/Tak-Au/Vehicle-Detection/blob/master/image0001.png, width=200))

I used opencv to read all images and store the image as X_data.  I assign 0 for non-car and 1 for car and store in y_data.  
I also augment the images by flipping vertical and horizontal.  I also augment by resizing the image.(Cell 3)    
Once all the images and their class are stored in X_data, and y_data, I used sklearn to split the data to training set and testing set (80% and 20% respectively). 

I train the model using the training set via 2 epochs.  I also set validation to .2 so that 20% of the training set will be used as validation so that I can see if the model is overtrained.


