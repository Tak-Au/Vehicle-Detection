# Vehicle-Detection

# The goals of this project are the following:
1.  Create and train a deep learning model to classify if patch image is either a car or non-car.
2.  Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
3. Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
4.  Estimate a bounding box for vehicles detected.

# Create and train deep learning model to classify car vs non-car:
To create a deep learning model, I used keras and transfering learning.  

First I start with a base model from Xception(Cell 6). The base model includes all the convolution layers.  Then I added GlobalMaxPooling2D layer and (2) dense layers.  The last dense layer has only 1 neuron with simgoid function.  This will allow the model to generate percentage if the image is non-car (0) or car (1).  

Once the model is built.  I train the model with the dataset.
The dataset has images (64X64) of car and non-car.  The non-car may include road, trees, guardrails.etc.  
I used opencv to read all images and store the image as X_data.  I assign 0 for non-car and 1 for car and store in y_data.  
I also augment the images by flipping vertical and horizontal.  I also augment by resizing the image.(Cell 3)    
