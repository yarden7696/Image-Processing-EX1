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
        img_rgb = img_rgb / 255.0
        return img_rgb
    else:
        img_grayscale=cv2.imread(filename,cv2.IMREAD_GRAYSCALE) # grayscale
        img_grayscale=img_grayscale/255.0
        # img_grayscale = cv2.cvtColor(img_grayscale, cv2.COLOR_BGR2GRAY)
        # img_grayscale = img_grayscale / np.max(img_grayscale)
        return img_grayscale
        pass




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
        pass




def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    RGB2YIQ = np.array([[0.299, 0.587, 0.114],[0.596, -0.275, -0.321],[0.212, -0.523, 0.311]])
    res_YIQ = np.dot(imgRGB, RGB2YIQ.T.copy())
    return res_YIQ
    pass



def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """

    RGB2YIQ = np.array([[0.299, 0.587, 0.114],[0.596, -0.275, -0.321],[0.212, -0.523, 0.311]])
    temp = np.linalg.inv(RGB2YIQ)
    res_BGR = np.dot(imgYIQ, temp.T.copy())
    return res_BGR
    pass



def display_img(img: np.ndarray):
    plt.gray()  # grayscale
    plt.imshow(img)
    plt.show()
    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """


    isRGB = bool(imgOrig.ndim == 3)  # case RGB image
    if isRGB:
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgOrig = np.copy(imgYIQ[:, :, 0])  # Y channel of the YIQ image
    else:
        imgYIQ=imgOrig # case grayscale

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





    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):


    isRGB = bool(imOrig.ndim == 3)  # case RGB image
    if isRGB:
        imgYIQ = transformRGB2YIQ(imOrig)
        imOrig = np.copy(imgYIQ[:, :, 0])  # Y channel of the YIQ image
    else:
        imgYIQ = imOrig  # case grayscale


    if np.amax(imOrig) <= 1 :
        imOrig = imOrig * 255  # normalized
    imOrig = imOrig.astype('uint8')

    histORGN, bins = np.histogram(imOrig, 256, [0, 255])

    # find the boundaries
    size = int(255 / nQuant)  # Divide the intervals evenly
    _Z = np.zeros(nQuant + 1, dtype=int)  # z is an array that will represents the boundaries
    for i in range(1, nQuant):
        _Z[i] = _Z[i - 1] + size
    _Z[nQuant] = 255  # The left border will always start at 0 and the right border will always end at 255

    _Q = np.zeros(nQuant)  # _Q is an array that represent the values of the boundaries

    quantized_lst = list()
    MSE_lst = list()

    for i in range(nIter):
        _newImg = np.zeros(imOrig.shape)

        for j in range(len(_Q)):  # j is a cell
            if j == len(_Q) - 1:
                right_cell = _Z[j + 1] + 1
            else:
                right_cell = _Z[j + 1]
            range_cell = np.arange(_Z[j], right_cell)
            _Q[j] = np.average(range_cell, weights=histORGN[_Z[j]:right_cell])
            check = np.logical_and(imOrig >= _Z[j], imOrig < right_cell)
            _newImg[check] = _Q[j]

        imOr=imOrig / 255.0
        imNew= _newImg / 255.0
        pixels = imOr.size
        subtract = np.subtract(imNew, imOr)
        sum_pixels = np.sum(np.square(subtract))
        MSE = np.sqrt(sum_pixels) / pixels
        MSE_lst.append(MSE)

        if isRGB:
            _newImg=_newImg / 255.0
            imgYIQ[:, :, 0] = _newImg
            _newImg = transformYIQ2RGB(imgYIQ)
        quantized_lst.append(_newImg)

        for b in range(1, len(_Z) - 1):  # b is boundary
            _Z[b] = (_Q[b - 1] + _Q[b]) / 2
        if len(MSE_lst) >= 2:
            if np.abs(MSE_lst[-1] - MSE_lst[-2]) <= 0.000001:
                break

    plt.plot(MSE_lst)
    plt.show()
    return quantized_lst, MSE_lst
    pass

