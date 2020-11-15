import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from PIL import Image

from preprocessing import StandardScaler

PICT_DIR = 'train_pictures/'  # Directory to save pictures
EVAL_DIR = 'inference_pictures/'  # Directory to save picture to evaluation


def files_to_pictures(file):
    """
    Function transform files path to image and save it to PICT_DIR
    :param file: PATH type of list of files
    :return:
    """
    scaler = StandardScaler()
    for indx in range(len(file)):
        data = genfromtxt(file[indx], delimiter=',')  # Load file
        data_scaled = scaler.fit_transform(data)  # Scale file

        fig = plt.figure(figsize=(0.5, 0.5))
        plt.plot(data_scaled[:, 0], data_scaled[:, 1], linewidth=2)  # Plot data
        fig.savefig(PICT_DIR + file[indx].name.split('.')[0])  # Save plot
        plt.close(fig)
    return 'Dataset is done'


def inference_file_to_picture(file_name):
    """
    Transform path of file to picture in EVAL_DIR directory
    :param file_name: String of path to file
    :return: Directory to file in .png format
    """
    scaler = StandardScaler()
    data = genfromtxt(file_name, delimiter=',')  # Load data
    data_scaled = scaler.fit_transform(data)  # Scale data
    fig = plt.figure(figsize=(0.36, 0.36))
    plt.plot(data_scaled[:, 0], data_scaled[:, 1], linewidth=2)  # Plot data
    file_name = EVAL_DIR + file_name.split('/')[-1].split('.')[0]  # Create new name
    fig.savefig(file_name)  # Save figure
    plt.close(fig)
    return file_name + '.png'


def data_to_picture(data):
    """
    Transform data to picture in EVAL_DIR directory
    :param data: Numpy matrix of data
    :return:
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)  # Scale data
    fig = plt.figure(figsize=(0.36, 0.36))
    plt.plot(data_scaled[:, 0], data_scaled[:, 1], linewidth=2)  # Plot data
    file_name = EVAL_DIR + 'inference_image'  # Create new name
    fig.savefig(file_name)  # Save figure
    plt.close(fig)
    return file_name + '.png'


def dataset_pictures(file):
    """
    Load all picture in one dataset
    :param file: PATH type of list of files
    :return: Dataset
    """
    X = np.zeros((len(file), 36, 36))
    for indx in range(len(file)):
        # Load image in grayscale
        img = np.array(Image.open(PICT_DIR+file[indx].name.split('.')[0]+'.png').convert('L'))
        X[indx] = img  # Save image in to dataset matrix

    X = np.array(list(map(trans, X)))  # Apply transformation to matrix of images
    return X


def inference_picture(file_dir):
    """
    Load inference picture to make inference through model
    :param file_dir: Path from inference_file_to_picture function
    :return: Picture matrix
    """
    pic = np.array(Image.open(file_dir).convert('L')).astype(float)
    pic = trans(pic).reshape(1, -1)
    return pic


def trans(pic):
    """
    Transform loaded picture
    :param pic: Picture from PIL
    :return: Flat matrix
    """
    pic[np.where(pic == 255)] = -1
    pic[np.where(pic != -1)] = 255
    pic[np.where(pic == -1)] = 0
    return pic.ravel()
