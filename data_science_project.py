# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
import tarfile
import os
from zipfile import ZipFile

with ZipFile('small_folder.zip') as zipObj:
  zipObj.extractall('/sample')

with ZipFile('/test.zip') as zipObj1:
  zipObj1.extractall('/test')

train_ds = ImageFolder('/sample/small_folder', transform=ToTensor())

test_ds = ImageFolder('/test', transform=ToTensor())

batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)

test_loader = DataLoader(test_ds, batch_size*2, shuffle=True)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

def accuracy_score(out, labels):
  _, preds = torch.max(out, dim=1)
  correct_preds = torch.sum(preds==labels).item()
  total_preds = len(preds)
  accuracy = torch.tensor(correct_preds/total_preds)
  return accuracy

num_epochs = 10
learning_rate = 0.0003
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

import os
data_dir = './data/cifar10'
classes = os.listdir(data_dir + "/train")

def predict_image(img, model):
    img = img.unsqueeze(0).to(device)
    out = model(img)
    _, preds  = torch.max(out, dim=1)
    return train_ds.classes[preds[0].item()]

import matplotlib.pyplot as plt
img, label = test_ds[8367]
plt.imshow(img.permute(1, 2, 0))
print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = "/sample/small_folder/n01440764/n01440764_10026.JPEG"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'test_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

