import json
import os

import numpy as np
from ultralytics import YOLO
import sahi.predict
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.postprocess.combine import GreedyNMMPostprocess
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from tqdm import tqdm

# Evaluates the trained model with per-class metrics:
# 1) slice level, on the validation and on the held-out test slices (Ultralytics validator)
# 2) trap level: whole pictures are scored exactly as in predict_sahiSlice.py and the
#    detections are matched to the original annotations.
#    a) the confidence threshold is chosen on the validation traps (THRESHOLDS sweep)
#    b) stress test: the held-out test traps are scored once, at the chosen threshold.
# All the results are written as csv files in OutputPath, the Ultralytics plots in its val/ and test/ subfolders.

#Setup your project variables:
# PLEASE EDIT THESE:

ProjectPath = '/path/to/main/folder/'

DatasetPath = ProjectPath + 'YOLOdataset/data.yaml'  # written by fillDatasetTXT.py

TrapSplitPath = ProjectPath + 'YOLOdataset/TrapSplit.csv'  # written by fillDatasetTXT.py

CocoPath = '/path/to/repo/dataset/Traps.json'  # original (not sliced) annotations, in this repository

CocoImagesPath = '/path/to/repo/dataset/Images/'

PretrainedModel = '/path/to/repo/model/best.pt'

OutputPath = ProjectPath + 'evaluation/'

# same settings as predict_sahiSlice.py
slice_height = 640
slice_width = 640
overlap_height_ratio = 0.6
overlap_width_ratio = 0.6

# confidence thresholds tried on the validation traps; the one with the highest mean F1
# over the TARGETS classes is used on the test traps (set it as tresh in predict_sahiSlice.py;
# 0.6 for the released model)
THRESHOLDS = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
TARGETS = ['Lobesia', 'Crypto']

IOU_MATCH = 0.5  # a detection is correct if its box overlaps an annotation with IoU >= IOU_MATCH

device = 'cuda:0'

os.makedirs(OutputPath, exist_ok=True)

model = YOLO(PretrainedModel)
names = model.names  # {0: 'Lobesia', 1: 'Crypto', 2: 'Other'}


def f1(p, r):
  return 2 * p * r / (p + r) if p + r > 0 else 0.0


## 1) Slice level metrics

with open(OutputPath + 'slice_metrics.csv', 'w') as f:
  f.write('split,target,class,precision,recall,f1,mAP50,mAP50-95\n')
  for split in ['val', 'test']:
    metrics = model.val(data=DatasetPath, split=split, device=device, plots=True,
                        project=OutputPath, name=split, exist_ok=True)
    for target, result in [('box', metrics.box), ('mask', metrics.seg)]:
      for i, c in enumerate(metrics.ap_class_index):
        p, r, ap50, ap = result.class_result(i)
        f.write(f'{split},{target},{names[c]},{p:.4f},{r:.4f},{f1(p, r):.4f},{ap50:.4f},{ap:.4f}\n')
      p, r, ap50, ap = result.mean_results()
      f.write(f'{split},{target},all,{p:.4f},{r:.4f},{f1(p, r):.4f},{ap50:.4f},{ap:.4f}\n')


## 2) Trap level metrics

def boxIoU(a, b):
  # a: (N, 4), b: (M, 4) in xyxy
  tl = np.maximum(a[:, None, :2], b[None, :, :2])
  br = np.minimum(a[:, None, 2:], b[None, :, 2:])
  inter = np.prod(np.clip(br - tl, 0, None), axis=2)
  areaA = np.prod(a[:, 2:] - a[:, :2], axis=1)
  areaB = np.prod(b[:, 2:] - b[:, :2], axis=1)
  return inter / (areaA[:, None] + areaB[None, :] - inter)


def readTraps(setName):
  with open(TrapSplitPath) as f:
    f.readline()
    return [line.split(',')[0] for line in f if line.split(',')[1] == setName]


with open(CocoPath) as f:
  coco = json.load(f)

classIds = {name: c for c, name in names.items()}
nc = len(names)


class BoxOnlyModel(UltralyticsDetectionModel):
  # The matching only uses boxes, so the masks are dropped: sahi merges overlapping slice
  # detections by their boxes anyway, and merging the masks can fail with
  # "ValueError: Invalid segmentation mask." at low confidence thresholds.
  @property
  def has_mask(self):
    return False


class NoMerge:
  # makes get_sliced_prediction return the detections of all the slices without merging them
  def __init__(self, **kwargs):
    pass

  def __call__(self, predictions):
    return predictions


sahi.predict.POSTPROCESS_NAME_TO_CLASS['NONE'] = NoMerge

# the merging done by get_sliced_prediction in predict_sahiSlice.py. It is class agnostic: each insect is seen in
# several overlapping slices, sometimes with different labels, and keeps only the label of its most confident one
merge = GreedyNMMPostprocess(match_threshold=0.5, match_metric='IOS', class_agnostic=True)


def predictTraps(traps, description):
  # Runs the model once per trap, at the lowest threshold, and keeps the unmerged slice detections.
  # Running at a higher threshold is the same as dropping the detections below it and then merging,
  # because sahi drops them on each slice before merging: evaluateTraps does that for every threshold.
  detection_model = BoxOnlyModel(
    model_path=PretrainedModel,
    confidence_threshold=min(THRESHOLDS),
    device=device
  )
  trapData = {}
  for trap in tqdm(traps, desc=description, unit='trap'):
    image = next(image for image in coco['images'] if os.path.splitext(image['file_name'])[0] == trap)
    annotations = [a for a in coco['annotations'] if a['image_id'] == image['id']]
    trueBoxes = np.array([[x, y, x + w, y + h] for x, y, w, h in (a['bbox'] for a in annotations)]).reshape(-1, 4)
    trueClasses = np.array([a['category_id'] - 1 for a in annotations], dtype=int)

    result = get_sliced_prediction(
      read_image(CocoImagesPath + image['file_name']),
      detection_model,
      slice_height=slice_height,
      slice_width=slice_width,
      overlap_height_ratio=overlap_height_ratio,
      overlap_width_ratio=overlap_width_ratio,
      perform_standard_pred=False,  # as in predict_sahiSlice.py: slices only, no extra pass on the shrunk whole picture
      postprocess_type='NONE',
      verbose=0
    )
    trapData[trap] = (trueBoxes, trueClasses, result.object_prediction_list)
  return trapData


def evaluateTraps(trapData, threshold):
  # returns the confusion matrix (rows: annotated class, columns: predicted class; the last
  # row/column is background) and, for each trap, the true and predicted count of each class
  confusion = np.zeros((nc + 1, nc + 1), dtype=int)
  trapCounts = {}
  for trap, (trueBoxes, trueClasses, slicePredictions) in trapData.items():
    predictions = [p for p in slicePredictions if p.score.value >= threshold]
    if len(predictions) > 1:
      predictions = merge(predictions)
    predictions = sorted(predictions, key=lambda p: -p.score.value)
    predBoxes = np.array([p.bbox.to_xyxy() for p in predictions]).reshape(-1, 4)
    predClasses = np.array([classIds[p.category.name] for p in predictions], dtype=int)

    # greedy class-agnostic matching, highest confidence first
    iou = boxIoU(predBoxes, trueBoxes)
    matchedTrue = np.zeros(len(trueBoxes), dtype=bool)
    for i in range(len(predBoxes)):
      candidates = np.where(~matchedTrue & (iou[i] >= IOU_MATCH))[0]
      if len(candidates):
        j = candidates[np.argmax(iou[i, candidates])]
        matchedTrue[j] = True
        confusion[trueClasses[j], predClasses[i]] += 1
      else:
        confusion[nc, predClasses[i]] += 1
    for j in np.where(~matchedTrue)[0]:
      confusion[trueClasses[j], nc] += 1

    trapCounts[trap] = [(np.sum(trueClasses == c), np.sum(predClasses == c)) for c in names]
  return confusion, trapCounts


def classMetrics(confusion):
  # per class: annotated, predicted, TP, FP, FN, precision, recall, f1
  rows = {}
  for c in names:
    tp = confusion[c, c]
    fp = confusion[:, c].sum() - tp  # false detections and other classes predicted as c
    fn = confusion[c, :].sum() - tp  # missed and c predicted as another class
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    rows[names[c]] = (confusion[c, :].sum(), confusion[:, c].sum(), tp, fp, fn, p, r, f1(p, r))
  return rows


def countError(annotated, predicted):
  return (predicted - annotated) / annotated if annotated else 0.0


## 2a) Confidence threshold sweep on the validation traps

validationData = predictTraps(readTraps('Validation'), 'Validation traps')
bestThreshold, bestScore = None, -1.0
with open(OutputPath + 'threshold_sweep.csv', 'w') as f:
  f.write('threshold,class,annotated,predicted,TP,FP,FN,precision,recall,f1,count_error\n')
  for threshold in THRESHOLDS:
    rows = classMetrics(evaluateTraps(validationData, threshold)[0])
    for name, (annotated, predicted, tp, fp, fn, p, r, f1score) in rows.items():
      f.write(f'{threshold},{name},{annotated},{predicted},{tp},{fp},{fn},{p:.4f},{r:.4f},{f1score:.4f},'
              f'{countError(annotated, predicted):.4f}\n')
    score = np.mean([rows[name][7] for name in TARGETS])
    if score > bestScore:
      bestThreshold, bestScore = threshold, score
print(f'Chosen confidence threshold: {bestThreshold} (mean validation F1 of {", ".join(TARGETS)}: {bestScore:.4f})')


## 2b) Stress test on the held-out test traps, at the chosen threshold

confusion, trapCounts = evaluateTraps(predictTraps(readTraps('Test'), 'Test traps'), bestThreshold)

with open(OutputPath + 'trap_counts.csv', 'w') as f:
  f.write('trap,' + ','.join(f'{names[c]}_true,{names[c]}_predicted' for c in names) + '\n')
  for trap, counts in trapCounts.items():
    f.write(trap + ',' + ','.join(f'{true},{predicted}' for true, predicted in counts) + '\n')

labels = [names[c] for c in names] + ['background']
with open(OutputPath + 'trap_confusion_matrix.csv', 'w') as f:
  f.write('true\\predicted,' + ','.join(labels) + '\n')
  for row, label in zip(confusion, labels):
    f.write(label + ',' + ','.join(str(v) for v in row) + '\n')

with open(OutputPath + 'trap_metrics.csv', 'w') as f:
  f.write(f'# confidence threshold {bestThreshold}\n')
  f.write('class,annotated,predicted,TP,FP,FN,precision,recall,f1,count_error\n')
  for name, (annotated, predicted, tp, fp, fn, p, r, f1score) in classMetrics(confusion).items():
    f.write(f'{name},{annotated},{predicted},{tp},{fp},{fn},{p:.4f},{r:.4f},{f1score:.4f},'
            f'{countError(annotated, predicted):.4f}\n')

for file in ['slice_metrics.csv', 'threshold_sweep.csv', 'trap_metrics.csv', 'trap_confusion_matrix.csv', 'trap_counts.csv']:
  print('\n' + file)
  print(open(OutputPath + file).read())
