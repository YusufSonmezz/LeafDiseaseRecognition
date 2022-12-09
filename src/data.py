"""
Data.py works for two components:
1 -> Preparing data directions
2 -> Preparing labels of the data
"""

from constant import *
import os
import glob

def getPathListAndLabelsOfPlants(directionOfDataset:str):
    """
    This functions returns path lists and labels of the images.
    :param directionOfDataset: direction of dataset that we want to process.
    :return:
    """

    file_path_list = glob.glob(os.path.join(directionOfDataset, "*"))

    path_list = []
    labeled_list = []
    for iter, file in enumerate(file_path_list):
        for path in glob.glob(os.path.join(file, "*")):
            path_list.append(path)
            labeled_list.append(iter) if directionOfDataset == TRAIN_DIR else []

    return path_list, labeled_list if directionOfDataset == TRAIN_DIR else []

def getLabeledListAsDictionary(directionOfDataset:str):
    """

    :param path_list:
    :return:
    """
    # Finds the length of direction -> "..\\dataset\\Parsed Dataset\\train" = len(dir) is 4
    # We find this variable to find the plant and disease name in below.
    wordsToPlantName = len(directionOfDataset.split("\\"))

    splittedPlantNames = {}
    for iter, folderPath in enumerate(glob.glob(os.path.join(directionOfDataset, "*"))):
        splittedPlantNames[iter] = folderPath.split("\\")[wordsToPlantName]

    return splittedPlantNames

def findEquivalentOfLabels(path_list:list,
                           labeledListAsDictionary:dict,
                           directionOfDataset:str):
    """
    Finds Equivalent of labels of folder names that is given.
    For example, in the valid folder, plants and their disease may not be the same as train folder.
    This function provides to know which category valid or test folder names belong to.
    :param path_list: list of str / list of every file in the folders
    :param labeledListAsDictionary: dict / Main label for finding the equivalent of file names
    :param directionOfDataset : str / is it a valid folder or test folder?
    :return:
    list of labels that one-to-one with list of images
    """
    # Finds the length of direction -> "..\\dataset\\Parsed Dataset\\train" = len(dir) is 4
    # We find this variable to find the plant and disease name in below.
    wordsToPlantName = len(directionOfDataset.split("\\"))

    labelOfFolder = []
    for iter, path in enumerate(path_list):
        plantAndDiseaseName = path.split("\\")[wordsToPlantName]
        labelOfFolder.append([key for key, value in labeledListAsDictionary.items() if value == plantAndDiseaseName].pop())
    return labelOfFolder




if __name__ == '__main__':

    path_list, label = getPathListAndLabelsOfPlants(TRAIN_DIR)
    #print(path_list[0])
    #print(len(path_list))

    splitted = getLabeledListAsDictionary(TRAIN_DIR)
    print(splitted)

    # Valid folder
    valid_path_list, _ = getPathListAndLabelsOfPlants(VALID_DIR)
    print(valid_path_list[2])
    print("len of valid path list..: ", len(valid_path_list))

    # Labels of valid folder
    labelsOfValid = findEquivalentOfLabels(path_list=valid_path_list, labeledListAsDictionary=splitted, directionOfDataset=VALID_DIR)
    #print(labelsOfValid)







