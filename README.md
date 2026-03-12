# Lobesia and Crypto finder tool
This repository contains the code, weights and metadata for the trained YOLO model described in the paper IN PREP. The system is capable of recognizing and counting insects attached to sticky traps using [Ultralytics YOLO](https://docs.ultralytics.com/it/) . It is specifically designed to recognize Crypto and Lobesia, and distinguish them from other insects stuck on the traps.

This readme describes the process we followed to train and test the model.

## Training

### Preparation of training data
To start, traps needed to be manually segmented to recognize the presence of the species you want to identify.
Segmentation of the pictures was to be done with a standalone software. We used [coco annotator](https://github.com/jsbroks/coco-annotator) for this project, but any other may work. The data was then exported in COCO format. The coco format contains the annotation information into a standardized json file, together with a folder containing all the images.

### Slicing the images
Recognizing the different insects require the resolution to be quite high, as the details are needed to distinguish different species. However most computers will be unable to analyze the images at full resolution as wholes, without any rescaling. For this reason, in this project we decided to use the [sahi](https://github.com/obss/sahi) tool. This was used for both training and then for images segmentation. The script `sliceCoco.py` was used to split images in partially overlapping sections. 

### Converting to YOLO format
Before proceeding with training, the dataset was be converted to Ultralytics YOLO standard using `coco2yolo.py` script. The pictures have then been moved manually to the newly created folder. Then, we generated the training/validation split files using the `fillDatasetTXT.py` script.

The model was then trained using the `ModelTraining.py` script. 

## Usage
The script `predict_sahiSlice.py` generate videos showing the incrementing count of detected targets. This is of course just an example, as the data can be used in many other ways.

## Model and metadata
the model weights are included in the `model` folder. The metadata instead are contained in the `metadata` subfolder, and include the training parameters, the training metrics, the model architecture.
