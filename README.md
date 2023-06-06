# vision-search-navigation
Repo for the walking robot's vision based navigation code, linked with paper "Real-time Vision-based Navigation for a Legged Robot in an Indoor Environment using High-level Interaction"



## Getting started

This repository features all the offline code used to predict a navigation path. To get started, I highly recommend looking through the code listed in notebooks/staticSegmentedNavigation.ipynb

To try out this code, please copy the **staticSegmentedNavigation.ipynb** notebook to your Google Colab and add an input image from the dataset listed below. 

Note: input images are of format: frame_0001.png

You will need to upload the image to your Google drive, mount it in Colab  and then modify the image path in the notebook to your Google drive. Once the file is linked properly, all cells should go through. You will see output data in the same Google drive folder where you added the image.



## Dataset

Data for evaluation is available below

Labeled Data: \href{https://drive.google.com/drive/folders/1sxXblBL04injdSfNBE3NMQGg7dtDwJAs?usp=sharing}{https://drive.google.com/drive/folders/1sxXblBL04injdSfNBE3NMQGg7dtDwJAs?usp=sharing}

Unlabeled Data: \href{https://drive.google.com/drive/folders/1xe9N7UEEH2GSFKTQ7Z9DMfBM1-183ovO?usp=sharing}{https://drive.google.com/drive/folders/1xe9N7UEEH2GSFKTQ7Z9DMfBM1-183ovO?usp=sharing}



## Offline code

The code to process images in a loop is also being shared via 3 files: vision.py to capture images, search.py that defines A* and UCS search methods as well as other helpful functions to conduct search, while navigation.py loads captured images, performs segmentation and search, and finally saves the output image.
