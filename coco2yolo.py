from ultralytics.data.converter import convert_coco

# Converts the sliced COCO dataset (written by sliceCoco.py) to the Ultralytics YOLO
# segmentation format. The slice pictures then have to be moved to the images/ folder
# of the new dataset, before running fillDatasetTXT.py.

#Setup your project variables:
# PLEASE EDIT THESE:

cocopath = '/path/to/slicedCOCOdataset/'
yolopath = '/path/to/YOLOdataset/'

convert_coco(labels_dir=cocopath, save_dir=yolopath, use_segments=True, use_keypoints=False, cls91to80=False)
