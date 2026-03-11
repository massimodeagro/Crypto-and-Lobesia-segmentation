from ultralytics.data.converter import convert_coco

cocopath = '/path/to/SLICEDCOCOdataset/'
yolopath = '/path/to/YOLOdataset/'

convert_coco(labels_dir=cocopath, save_dir=yolopath, use_segments=True, use_keypoints=False, cls91to80=False)