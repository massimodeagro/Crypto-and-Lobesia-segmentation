import os
import random

percentageVAL = 0.1

ProjectPath = '/path/to/main/folder/YOLOdataset/'

ImagesPath = ProjectPath + 'images/'

images = os.listdir(ImagesPath)

random.shuffle(images)

splitindex = int(len(images) * percentageVAL)

training = images[splitindex:]
validation = images[:splitindex]

with open(ProjectPath + "Training.txt", "w") as f:
  for name in training:
    f.write(ImagesPath + name + '\n')
    
with open(ProjectPath + "Validation.txt", "w") as f:
  for name in validation:
    f.write(ImagesPath + name + '\n')
