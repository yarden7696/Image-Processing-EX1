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

    # im = cv2.imread(filename, representation-1)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im = im/np.max(im)
    #
    # return im

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
    # if (imgRGB.ndim == 3):  # if the image is in rgb colors
    #     row = len(imgRGB)
    #     col = len(imgRGB[0])
    #     z = len(imgRGB[0][0])
    #     array = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    #     rgb_trans = imgRGB.transpose()  # transpose the matrix for multiply it
    #     yiq_mul = array.dot(rgb_trans.reshape(3, row * col))  # the matrix after multiply
    #     yiq_before = np.reshape(yiq_mul, (z, col, row))  # yiq before reshape
    #     yiq = np.transpose(yiq_before)  # the original yiq
    #     return yiq
    # else:  # if the image is in grayscale
    #     return imgRGB

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
    rgb2yiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    yiq2rgb = linalg.inv(rgb2yiq)  # reverse matrix to multiply
    imgRGB = img.dot(yiq2rgb)
    m, n, z = np.shape(imgRGB)

    for i in range(m):  # imdexes that not in range [0,1]
        for j in range(n):
            for t in range(z):
                if imgRGB[i, j, t] < 0:
                    imgRGB[i, j, t] = 0

                elif imgRGB[i, j, t] > 1:
                    imgRGB[i, j, t] = 1

    return imgRGB



def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    if (imgOrig.ndim == 2):  # if the image is in grayscale
        imgOrig = cv2.normalize(imgOrig.astype('float64'), None, 0, 255, cv2.NORM_MINMAX)
        y_channel = imgOrig

    else:  # if the image is in rgb colors
        imgOrig = cv2.normalize(imgOrig.astype('float64'), None, 0, 255, cv2.NORM_MINMAX)
        y_channel = transformRGB2YIQ(imgOrig)[:, :, 0]  # take the y channel

    histogram = calHist(y_channel)  # step 1 - calculate the histogram image
    cum_sum = calCumSum(histogram)  # step 2 - calculate the cumulative sum
    nor_cum_sum = cum_sum / cum_sum.max()  # step 3 - normalazied the cum sum
    map_img = nor_cum_sum * 255  # step 4 - mapping the old intensity colors to new intensity color
    round_map = map_img.astype('uint8')  # step 5 - round the values
    old_y_channel = np.array(y_channel).astype('uint8')  # casting to int for set each value at the round_map
    new_img = round_map[old_y_channel]  # step 6 - set the new intensity value according to the map
    imgOrig = cv2.normalize(imgOrig.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    histogram_new = calHist(new_img)  # the new image's histogram

    if (imgOrig.ndim == 2):
        return new_img, histogram, histogram_new

    else:
        yiq = transformRGB2YIQ(imgOrig)
        yiq[:, :, 0] = new_img / 255
        rgb = transformYIQ2RGB(yiq)
        return rgb, histogram, histogram_new


def calHist(
        img: np.ndarray) -> np.ndarray:  # side function that calculate the number each intensity color and return an array
    img_flat = img.ravel()  # flat the array for running in one for
    hist = np.zeros(256)
    for pix in img_flat:
        pix = math.floor(pix)
        hist[pix] += 1

    return hist


def calCumSum(arr: np.ndarray) -> np.ndarray:
    cum_sum = np.zeros_like(arr)
    cum_sum[0] = arr[0]
    for i in range(1, len(arr)):
        cum_sum[i] = cum_sum[i - 1] + arr[i]

    return cum_sum


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
