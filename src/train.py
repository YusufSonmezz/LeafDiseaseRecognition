from data import *
from constant import *
from preprocess import tensorize_image, one_hot_encoder
import numpy as np
from model_ResNet import ResNetModel
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from utils import *

################## Parameters ###############
cuda = True
batch_size = 16
epochs = 6
numberOfClasses = len(getLabeledListAsDictionary(TRAIN_DIR))
save_best_model = SaveBestModel()
#############################################

# Labels of images are numeric. This dictionary provides to know equivalent of number.
LabelDashboardAsDict = getLabeledListAsDictionary(TRAIN_DIR)

# # Getting all image paths and their labels
# Train
image_path_list, labelOfDataset = getPathListAndLabelsOfPlants(TRAIN_DIR)
# Valid
valid_path_list, _ = getPathListAndLabelsOfPlants(VALID_DIR)
labelsOfValid = findEquivalentOfLabels(path_list=valid_path_list, labeledListAsDictionary=LabelDashboardAsDict, directionOfDataset=VALID_DIR)

# Generates random numbers between 0 and length of image_path_list to shuffle array.
indicesTrain = np.random.permutation(len(image_path_list))
indicesValid = np.random.permutation(len(valid_path_list))

# # Shuffles arrays.
# Train
image_path_list = list(np.array(image_path_list)[indicesTrain])
labelOfDataset = list(np.array(labelOfDataset)[indicesTrain])
# Valid
valid_path_list = list(np.array(valid_path_list)[indicesValid])
labelsOfValid = list(np.array(labelsOfValid)[indicesValid])

# Define steps per epoch
steps_per_epoch = len(image_path_list)//batch_size

# Define Model and switch to train mode
ResNet50 = ResNetModel(50, numberOfClasses)
ResNet50.train()

# Define Loss Functions and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ResNet50.parameters(), lr=1e-4, weight_decay = 1e-6)

# Define Optimizer Schedule
...

# Cuda implementation
if cuda:
    ResNet50 = ResNet50.cuda()
    criterion = criterion.cuda()

# Values of the results
lossesTrain = AverageMeter('Loss', ':.4f')
lossesValid = AverageMeter('Valid Loss', ':.4f')
loss_values = []
accuracy_values = []

# Training
for epoch in range(epochs):
    running_loss = 0

    # Accuracy
    trainCorrect = 0
    validCorrect = 0

    trainTotal = 0
    validTotal = 0

    for ind in tqdm.tqdm(range(steps_per_epoch), ncols=100):
        # Prepares batchs of input and label paths
        batch_input_path_list = image_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label = torch.LongTensor(labelOfDataset[batch_size*ind:batch_size*(ind+1)]).cuda()

        # Joins the paths and tensorize data
        batch_input = tensorize_image(batch_input_path_list, (32, 32), cuda)

        # Cleans optimizer and model weights
        optimizer.zero_grad()
        ResNet50.zero_grad()

        # Fits model with data
        output = ResNet50(batch_input)

        # Prediction for the accuracy
        predictionTrain = prediction(output)

        # Loss Function
        # Evaluate the differences between prediction and real data.
        loss = criterion(output, batch_label)
        loss.backward()

        # Optimizer
        optimizer.step()

        # Assigns data to variables
        running_loss += loss.float()
        lossesTrain.update(loss.float(), batch_input.size(0))

        # Accuracy Train
        trainTotal += batch_label.size(0)
        trainCorrect += (predictionTrain == batch_label).sum().item()

        # VALIDATION PART
        if ind == steps_per_epoch - 1:
            # For validation loss graph and saving model
            validation_lossesLIST = []
            val_lossINT = 0

            # For evaluation of model
            ResNet50.eval()

            for (valid_input_path, valid_label) in zip(valid_path_list, labelsOfValid):
                # Joins the paths and tensorize data
                valid_input = tensorize_image([valid_input_path], (32, 32), cuda)
                valid_label = torch.LongTensor([valid_label]).cuda()

                with torch.no_grad():
                    # Prediction of model
                    output_ = ResNet50(valid_input)
                    vallLoss = criterion(output_, valid_label)

                # Prediction Valid
                predictionValid = prediction(output_)

                # Assign values to lists and int
                trainAccuracy = 100 * (trainCorrect / trainTotal)
                accuracy_values.append(trainAccuracy)
                validation_lossesLIST.append(vallLoss.float())
                lossesValid.update(vallLoss.float(), valid_input.size(0))
                val_lossINT += vallLoss.float()

                # Accuracy Valid
                validTotal += valid_label.size(0)
                validCorrect += (predictionValid == valid_label).sum().item()

            # Model returns to train
            ResNet50.train()

            val_lossINT = val_lossINT / len(valid_path_list)
            validAccuracy = 100 * (validCorrect / validTotal)

    # Saves the best model
    save_best_model(current_valid_loss=val_lossINT, epoch=epoch, model=ResNet50, criterion=criterion, optimizer=optimizer)

    # Train loss of epoch for plotting
    loss_values.append(running_loss / steps_per_epoch)

    # Printing all information to get an insight
    print("\nTrain Accuracy is ", trainAccuracy, "%")
    print("Valid Accuracy is ", validAccuracy, "%")
    print(lossesTrain)
    print(lossesValid)

# Saves last model and plots the loss graph
Plot(range(epochs), loss_values, ylabel="Loss", save = True)
Plot(range(len(valid_input)), validation_lossesLIST, ylabel="Val Loss", save=True)

torch.save(ResNet50, fr"C:\Users\Yusuf\Desktop\Tasarım\AI\models\{lossesTrain.avg}.pth")



























