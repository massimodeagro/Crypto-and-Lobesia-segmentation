import ultralytics
from ultralytics import YOLO
import os
import cv2
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from tqdm import tqdm
from multiprocessing import Pool

# Scores whole trap pictures with SAHI (the picture is cut into overlapping 640x640 slices,
# the model runs on each slice and the detections are merged back on the whole picture, regardless of class).
# For each picture it writes a video, detections_fewfps_<picture>.mp4, in which the
# detections appear one by one, with the running count of each class.

#Setup your project variables:
# PLEASE EDIT THESE:

ProjectPath = '/path/to/main/folder/'

ImagePath = ProjectPath + 'data/images/' #put here the pictures to score

PretrainedModel = '/path/to/repo/model/best.pt'

colors = {'Crypto': (255, 0, 0), 'Lobesia': (0, 255, 0), 'Other': (0, 0, 255)}  # BGR: blue, green, red

# the same settings are used by evaluateModel.py
slice_height = 640
slice_width = 640
overlap_height_ratio = 0.6
overlap_width_ratio = 0.6
tresh = 0.6  # chosen on the validation traps by evaluateModel.py

outvideofps=10  # frames per second of the output videos

detection_model = AutoDetectionModel.from_pretrained(
  model_type='ultralytics',
  model_path=PretrainedModel,
  confidence_threshold=tresh,
  device='cuda:0'
)

images = os.listdir(ImagePath)

def makeVideo(image):
  print(image)
  draw_image = cv2.imread(ImagePath+image)
  result = get_sliced_prediction(
    read_image(ImagePath+image),
    detection_model,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    # by default sahi also runs the model on the whole picture shrunk to 640 px, where insects are a few pixels
    # wide and dark clusters are detected as large non-target objects: only the slices are used
    perform_standard_pred=False,
    # each insect is seen in several overlapping slices, sometimes with different labels: merge them regardless
    # of class, so that it is counted once, with the label of its most confident detection
    postprocess_class_agnostic=True
  )

  detection_counts = {'Crypto': 0, 'Lobesia': 0, 'Other': 0, 'Total': 0}
  toshow = cv2.resize(draw_image, (0, 0), fx=1, fy=1)
  frame_width = int(toshow.shape[1])
  frame_height = int(toshow.shape[0])

  size = (frame_width, frame_height)

  # The video has the size of the picture and is saved in the current folder
  outvideo = cv2.VideoWriter('detections_fewfps_' + image.split('.')[0] + '.mp4',
                           cv2.VideoWriter_fourcc(*'MJPG'),
                           outvideofps, size)

  y_pos = 100
  for cat, count in detection_counts.items():
    color = colors.get(cat, (0, 0, 0))  # black for 'Total'
    cv2.putText(toshow, f"{cat}: {count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
    y_pos += 100

  for i in range(2*outvideofps):  # 2 s on the picture before any detection
    #cv2.imshow('t', toshow)
    outvideo.write(toshow)
    #cv2.waitKey(0)

  for pred in tqdm(result.object_prediction_list):
    x1,y1,x2,y2 = pred.bbox.to_voc_bbox()
    like = pred.score.value
    category = pred.category.name
    detection_counts[category] += 1
    detection_counts['Total'] += 1

    cv2.rectangle(draw_image, (x1,y1), (x2,y2), colors[category], 5)
    cv2.putText(draw_image, f"{category}: {like:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[category], 2)

    toshow = cv2.resize(draw_image, (0,0), fx=1, fy=1).copy()
    y_pos = 100
    for cat, count in detection_counts.items():
        color = colors.get(cat, (0, 0, 0))  # black for 'Total'
        cv2.putText(toshow, f"{cat}: {count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
        y_pos += 100
    for i in range(int(0.1 * outvideofps)):  # 0.1 s for each new detection
      #cv2.imshow('t', toshow)
      outvideo.write(toshow)
      #cv2.waitKey(0)

  for i in range(int(2 * outvideofps)):  # 2 s on the final count
    #cv2.imshow('t', toshow)
    outvideo.write(toshow)
    #cv2.waitKey(0)

  outvideo.release()

# one picture per process
with Pool() as p:
  p.map(makeVideo, images)
