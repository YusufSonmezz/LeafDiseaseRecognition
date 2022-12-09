from model_ResNet import ResNetModel
from data import getLabeledListAsDictionary, getPathListAndLabelsOfPlants
from constant import *
from preprocess import tensorize_image
from utils import prediction

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import glob
import os

############## Preparing the model ################
# Getting number of classes for creating the model
numberOfClasses = len(getLabeledListAsDictionary(TRAIN_DIR))

# Creating model and optimizer
ResNet50 = ResNetModel(50, numberOfClasses)
optimizer = optim.Adam(ResNet50.parameters(), lr=1e-4, weight_decay = 1e-6)

# Fitting model with pretrained weights
checkpoint = torch.load(BEST_MODEL_DIR)
ResNet50.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Turn model evaluation mode and cuda
ResNet50 = ResNet50.cuda()
ResNet50 = ResNet50.eval()
##################################################

############### Preparing Data ###################
# Getting test data path list
test_path_list = glob.glob(os.path.join(TEST_DIR, "*"))

# Finding the names of images
testImagesNames = []
for test_path in test_path_list:
        testImagesNames.append(test_path.split("\\")[-1].split(".")[0])

# Tensorize images
test_images_tensorized = tensorize_image(test_path_list, (32, 32), True)

# All labels
testLabels = getLabeledListAsDictionary(TRAIN_DIR)
##################################################

# Making predictions
with torch.no_grad():
        results = ResNet50(test_images_tensorized)

# Finding indices that has maximum value
results = prediction(results)

# Finding the equivalent of the results in labelDictionary
resultsClassNames = []
for result in results:
        for key, value in testLabels.items():
                if result == key:
                        resultsClassNames.append(value)

# Combain image names and results class in the same dictionary
ResultDictForDataFrame = {
        'Facts': testImagesNames,
        'Predictions': resultsClassNames
}

# To see and compare predictions and facts in the same table
comparingResults = pd.DataFrame(ResultDictForDataFrame)

# Adding new Row to see if there is equal or not
comparingResults["Match"] = np.where((comparingResults['Facts'] == comparingResults['Predictions']), True, False)

# Printing all the results
print(comparingResults)



