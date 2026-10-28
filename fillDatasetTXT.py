import json
import os
import random
import re
from collections import Counter

# Splits the sliced dataset by TRAP (original picture), not by slice, so that
# overlapping slices of the same trap never end up in different sets.
# NTEST traps are held out completely (final stress test), the remaining
# traps are divided between training and validation according to percentageVAL.
# Traps are stratified by their dominant target species so that every set
# contains both Lobesia and Crypto.
# Writes, in ProjectPath: Training.txt, Validation.txt and Test.txt (the slice lists),
# data.yaml (used by ModelTraining.py and evaluateModel.py) and TrapSplit.csv
# (set and number of insects of each class per trap).

#Setup your project variables:
# PLEASE EDIT THESE:

ProjectPath = '/path/to/main/folder/YOLOdataset/'

ImagesPath = ProjectPath + 'images/'

CocoPath = '/path/to/repo/dataset/Traps.json'  # original (not sliced) annotations, in this repository

percentageVAL = 0.1
NTEST = 4  # traps never seen during training, half Lobesia and half Crypto dominated
MIN_TARGET = 10  # minimum number of target insects for a trap to be picked as test trap
SEED = 0

names = {0: 'Lobesia', 1: 'Crypto', 2: 'Other'}


def trapName(sliceName):
  # sahi names slices as <trap>_<cocoImageId>_<x0>_<y0>_<x1>_<y1>.png
  return re.sub(r'_\d+_\d+_\d+_\d+_\d+\.png$', '', sliceName)


# Count insects per trap on the original annotations (slices overlap and would double count)
with open(CocoPath) as f:
  coco = json.load(f)
imageNames = {image['id']: os.path.splitext(image['file_name'])[0] for image in coco['images']}
counts = {name: Counter() for name in imageNames.values()}
for annotation in coco['annotations']:
  counts[imageNames[annotation['image_id']]][annotation['category_id'] - 1] += 1

slices = {}
for name in sorted(os.listdir(ImagesPath)):
  slices.setdefault(trapName(name), []).append(name)

missing = set(slices) - set(counts)
if missing:
  raise ValueError(f'Traps without annotations in {CocoPath}: {sorted(missing)}')


def stratum(trap):
  lobesia, crypto = counts[trap][0], counts[trap][1]
  if lobesia > crypto:
    return 'Lobesia'
  if crypto > lobesia:
    return 'Crypto'
  return 'Other'


rng = random.Random(SEED)
traps = sorted(slices)

# Final test traps
test = []
for species in ['Lobesia', 'Crypto']:
  candidates = [t for t in traps if stratum(t) == species and counts[t][0 if species == 'Lobesia' else 1] >= MIN_TARGET]
  test += rng.sample(candidates, NTEST // 2)

# Validation traps, stratified
remaining = [t for t in traps if t not in test]
validation = []
for group in ['Lobesia', 'Crypto', 'Other']:
  members = [t for t in remaining if stratum(t) == group]
  validation += rng.sample(members, round(len(members) * percentageVAL))
training = [t for t in remaining if t not in validation]

split = {'Training': training, 'Validation': validation, 'Test': test}

for setName, setTraps in split.items():
  with open(ProjectPath + setName + '.txt', 'w') as f:
    for trap in setTraps:
      for name in slices[trap]:
        f.write(ImagesPath + name + '\n')

with open(ProjectPath + 'TrapSplit.csv', 'w') as f:
  f.write('trap,set,slices,' + ','.join(names.values()) + '\n')
  for setName, setTraps in split.items():
    for trap in sorted(setTraps, key=lambda t: (len(t), t)):
      f.write(f'{trap},{setName},{len(slices[trap])},' + ','.join(str(counts[trap][c]) for c in names) + '\n')

with open(ProjectPath + 'data.yaml', 'w') as f:
  f.write(f'path: {ProjectPath}\ntrain: Training.txt\nval: Validation.txt\ntest: Test.txt\nnames:\n')
  for c, name in names.items():
    f.write(f'  {c}: {name}\n')

for setName, setTraps in split.items():
  total = Counter()
  for trap in setTraps:
    total += counts[trap]
  print(f'{setName}: {len(setTraps)} traps, {sum(len(slices[t]) for t in setTraps)} slices, '
        + ', '.join(f'{names[c]} {total[c]}' for c in names))
print('Test traps:', ', '.join(sorted(test, key=lambda t: (len(t), t))))
