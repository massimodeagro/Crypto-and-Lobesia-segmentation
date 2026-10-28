import ultralytics
from ultralytics import YOLO

# Trains the YOLO11 segmentation model on the sliced dataset.
# The training and validation slices come from different traps (see fillDatasetTXT.py);
# the test traps are not used here, only by evaluateModel.py.

ultralytics.checks()

#Setup your project variables:
# PLEASE EDIT THESE:

ProjectPath = '/path/to/base/folder/'


DatasetPath = ProjectPath + 'YOLOdataset/data.yaml'  # written by fillDatasetTXT.py

## To resume an interrupted training, load its last.pt instead and call model.train(resume=True)
#PretrainedModel = ProjectPath + 'last.pt'

# Start from the COCO-pretrained medium segmentation model
segmentation = 'yolo11m-seg.pt'


model = YOLO(segmentation)  # load a pretrained model

# Training stops when the validation fitness has not improved for `patience` epochs
# (the released model stopped after 180 epochs, the best weights are kept in best.pt)
EPOCHS = 1000
BATCH = 16  # lower it if the GPU runs out of memory

# Traps have no up/down orientation, so vertical flips and full rotations are valid augmentations.
# No class weighting or oversampling: all the other augmentations are left at their defaults.
trainingResults = model.train(data=DatasetPath, epochs=EPOCHS, imgsz=640, batch=BATCH, patience=100,
                              flipud=0.5, degrees=180, plots=True, device=0)
