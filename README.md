# IP_Task3
The Objective is to detect traffic signs in videos and images and recognise them, so as to concept can be implemented in real time.

We have approached to implement this in multiple ways.

1. Training FRCNN model for GTSDB (German Traffic sign recognition benchmark) Dataset, which has 43 classes, 
   Total fo 900 annoted images (divided in 600 training images and 300 evaluation images).
   
   ![alt text](https://www.researchgate.net/profile/Zhuonan-Hao/publication/340134486/figure/fig5/AS:872819097366531@1585107697984/A-sample-image-from-GTSDB.ppm)
   
   ![alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0893608018300054-gr6b.jpg)
   
   
   The challenge with this dataset is discrepency in classes, Data distribution is very uneven throughout the 43 classes, due to which model isnt performing well.
   Result for the same for images and videos can be found in the respective folder.


2. Training FRCNN Model for 8 classes
   We have manually collected data from different dataset and narrow it down to 8 basic classes as:
   'No parking No stopping'
   'No entry'
   'Pedestrian Crossing'
   'Yield'
   'Speed limit'
   'Keep right'
   'Priority road'
   'No passing veh over 3.5 tons'
   The result for this one is slightly better than former as it was better in terms of data distribution but for precise object detection, 
   it still had very low no of images (approx 80-100 per class), and thus results arent that good.
   Results are attached in the respective folder.
   
3. Using Recognition and detection model Together (DetectGeneralisedSign_Recognition):
   Here we comprised all the 43 classes of GTSDB together into one common class as traffic sign, and trained the model using ssd_inception_coco_v2 config for faster results.
   First we detect the sign in the frame, crop it and fed to Traffic sign recognition model, which we have trained using GTSRB (German Traffic sign recognition benchmark) 
   Dataset, which has 43 classes, 
   Training set contains 39209 labeled images and the test set contains 12630 images(cropped only).
   
   ![alt text](https://www.researchgate.net/profile/Wen_Lihua/publication/322945549/figure/fig1/AS:601782556295179@1520487550890/The-total-43-classes-in-GTSRB-From-top-to-bottom-there-are-four-categories.png)
   
   By far this approach for us is turned out to be better performing than the formers.
   Results for the video and images are in the resepective folder.
   
   (As we are operating on the cropped result obtained from the Detection model, we nneed to modify primary visualization_utils.py file 
   in utils directory(models/research/object_detection/utils/) the same modified file is attached in the folder.
   Morever, in the result video, the sign speed limit 30km/h is different(blue) from the one we have used to train our recognition model using GTSRB(red), which produces 
   different result than expected. The later red signs can be correctly identified by the model)
