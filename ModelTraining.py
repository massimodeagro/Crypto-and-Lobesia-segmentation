import ultralytics
from ultralytics import YOLO

ultralytics.checks()

#Setup your project variables:
# PLEASE EDIT THESE:

ProjectPath = '/path/to/base/folder'


DatasetPath = ProjectPath + 'dataSLICED/data.yaml'

## If present
#PretrainedModel = ProjectPath + 'last.pt'

# Load a model
segmentation = 'yolo11n-seg.pt'


model = YOLO(segmentation)  # load a pretrained model

# Use the model
EPOCHS = 200000 # you choose
trainingResults = model.train(data=DatasetPath, epochs=EPOCHS, plots=True, batch=48, patience=500)
