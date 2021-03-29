"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import linalg

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 207205972


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    if representation==2:
        img_rgb =cv2.imread(filename)
        img_rgb=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb / np.max(img_rgb)
        return img_rgb
    else:
        img_grayscale=cv2.imread(filename) # grayscale
        img_grayscale = cv2.cvtColor(img_grayscale, cv2.COLOR_BGR2GRAY)
        img_grayscale = img_grayscale / np.max(img_grayscale)
        return img_grayscale




def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    if representation==2:
        rgb_img=imReadAndConvert(filename,2)
        plt.imshow(rgb_img)
        plt.show()
    else:
        grayscale_img = imReadAndConvert(filename, 1)
        plt.imshow(grayscale_img, cmap='gray')
        plt.show()





def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    image = np.copy(imgRGB)
    RGB2YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    YIQ_image = image.dot(RGB2YIQ)
    r, c, d = np.shape(YIQ_image)
    r = range(r)
    c = range(c)
    d = range(d)
    for i in r:
        for j in c:
            for t in d:
                if YIQ_image[i, j, t] > 1:
                    YIQ_image[i, j, t] = 1

                elif YIQ_image[i, j, t] < 0:
                    YIQ_image[i, j, t] = 0
    return YIQ_image


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    img = np.copy(imgYIQ)
    RGB2YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    YIQ2RGB = np.linalg.inv(RGB2YIQ)
    imgRGB = img.dot(YIQ2RGB)
    r, c, d = np.shape(imgRGB)
    r = range(r)
    c = range(c)
    d = range(d)
    for i in r:
        for j in c:
            for t in d:
                if imgRGB[i, j, t] < 0:
                    imgRGB[i, j, t] = 0

                elif imgRGB[i, j, t] > 1:
                    imgRGB[i, j, t] = 1
    return imgRGB


def display_img(img: np.ndarray):
    plt.gray()  # case of grayscale
    plt.imshow(img)
    plt.show()
    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """

    #show_img(imgOrig) # display the input image

    isRGB = bool(imgOrig.ndim == 3)  # case RGB image
    if isRGB:
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgOrig = np.copy(imgYIQ[:, :, 0])  # Y channel of the YIQ image
    else:
        imgYIQ=imgOrig

    imgOrig = imgOrig * 255
    imgOrig = (np.around(imgOrig)).astype('uint8')

    histOrg, bin_edges = np.histogram(imgOrig.flatten(), 256, [0, 255])  # Calculate a histogram of the original image
    cumsum = histOrg.cumsum()  # calculate cumsum

    imgScale = np.ma.masked_equal(cumsum, 0)
    imgScale = (imgScale - imgScale.min()) * 255 / (imgScale.max() - imgScale.min())  # scale to our histogram
    after_scale = np.ma.filled(imgScale, 0).astype('uint8')  # check that all pixels are integers

    # after i made scale i need to map every point in cumsum to new point in the linear line
    imgEq = after_scale[imgOrig.astype('uint8')]
    histEQ, bin_edges2 = np.histogram(imgEq.flatten(), 256, [0, 256])  # Calculate a histogram of the new image

    if isRGB:  # RGB case
        imgEq = (imgEq / 255)
        imgYIQ[:, :, 0] = imgEq
        rgb_img = transformYIQ2RGB(imgYIQ)
        display_img(rgb_img)
    else:  # case grayscale
        display_img(imgEq)
    return imgEq, histOrg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
