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



def imReadAndConvert (filename: str, representation: int) -> np.ndarray:
    """
            Reads an image, and returns the image converted as requested
            :param filename: The path to the image
            :param representation: GRAY_SCALE or RGB
            :return: The image object
    """
    if representation == 2:  # RGB
        img_rgb = cv2.imread(filename)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb / 255.0  # normalization
        return img_rgb
    else:  # grayscale
        img_grayscale = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img_grayscale = img_grayscale/255.0  # normalization
        return img_grayscale
        pass




def imDisplay(filename: str, representation: int):
    """
            Reads an image as RGB or GRAY_SCALE and displays it
            :param filename: The path to the image
            :param representation: GRAY_SCALE or RGB
            :return: None
    """
    if representation == 2:  # RGB
        rgb_img = imReadAndConvert(filename, 2)
        plt.imshow(rgb_img)
        plt.show()
    else:  # grayscale
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
    RGB2YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    res_YIQ = np.dot(imgRGB, RGB2YIQ.T.copy())  # dot product of vectors from RGB2YIQ and imgRGB
    return res_YIQ
    pass



def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
        Converts an YIQ image to RGB color space
        :param imgYIQ: An Image in YIQ
        :return: A RGB in image color space
    """
    RGB2YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    temp = np.linalg.inv(RGB2YIQ)  # Compute the inverse of a matrix
    res_BGR = np.dot(imgYIQ, temp.T.copy())  # dot product of vectors from temp and imgYIQ
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
    isRGB = bool(imgOrig.ndim == 3)  # RGB
    if isRGB:
        imgYIQ = transformRGB2YIQ(imgOrig)  # Convert to YIQ

        imgOrig = np.copy(imgYIQ[:, :, 0])  # save Y channel
    else:
        imgYIQ=imgOrig  # grayscale

    imgOrig = imgOrig * 255
    # Rounding the number, copying the mat and cast the matrix values to integers
    imgOrig = (np.around(imgOrig)).astype('uint8')
    histOrg, bins = np.histogram(imgOrig.flatten(), 256, [0, 255])  # Calculate a histogram of the original image
    cumsum = histOrg.cumsum()  # calculate cumsum
    imgScale = np.ma.masked_equal(cumsum, 0)
    imgScale = (imgScale - imgScale.min()) * 255 / (imgScale.max() - imgScale.min())  # scale to our histogram
    after_scale = np.ma.filled(imgScale, 0).astype('uint8')  # cast the matrix values to integers

    # after i made scale i need to map every point in cumsum to new point in the linear line
    imgEq = after_scale[imgOrig.astype('uint8')]
    histEQ, bins2 = np.histogram(imgEq.flatten(), 256, [0, 256])  # Calculate a histogram of the new image

    if isRGB:  # RGB
        imgEq = (imgEq / 255)
        imgYIQ[:, :, 0] = imgEq
        rgb_img = transformYIQ2RGB(imgYIQ)  # Convert back to RGB
        display_img(rgb_img)

    else:  # grayscale
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
    isRGB = bool(imOrig.ndim == 3)  # RGB image
    if isRGB:
        imgYIQ = transformRGB2YIQ(imOrig)  # Convert to YIQ
        imOrig = np.copy(imgYIQ[:, :, 0])  # save Y channel
    else:  # grayscale
        imgYIQ = imOrig

    if np.amax(imOrig) <=1:  # its means that imOrig is normalized
        imOrig = imOrig * 255
    imOrig = imOrig.astype('uint8')  # cast the matrix values to integers

    # Calculate a histogram of the original image
    histORGN, bins = np.histogram(imOrig, 256, [0, 255])

    # find the boundaries
    size = int(255 / nQuant)  # Divide the intervals evenly
    _Z = np.zeros(nQuant + 1, dtype=int)  # _Z is an array that will represents the boundaries
    for i in range(1, nQuant):
        _Z[i] = _Z[i - 1] + size
    _Z[nQuant] = 255  # The left border will always start at 0 and the right border will always end at 255
    _Q = np.zeros(nQuant)  # _Q is an array that represent the values of the boundaries

    quantized_lst = list()
    MSE_lst = list()

    for i in range(nIter):
        _newImg = np.zeros(imOrig.shape)  # Initialize a matrix with 0 in the original image size

        for j in range(len(_Q)):  # every j is a cell
            if j == len(_Q) - 1:
                right_cell = _Z[j + 1] + 1
            else:
                right_cell = _Z[j + 1]
            range_cell = np.arange(_Z[j], right_cell)
            _Q[j] = np.average(range_cell, weights=histORGN[_Z[j]:right_cell])
            # mat is a matrix that is initialized in T / F. any value that satisfies the two conditions will get T, otherwise -F
            mat = np.logical_and(imOrig >= _Z[j], imOrig < right_cell)
            _newImg[mat] = _Q[j]  # Where there is a T we will update the new value

        imOr = imOrig / 255.0
        imNew = _newImg / 255.0
        MSE = np.sqrt(np.sum(np.square(np.subtract(imNew, imOr)))) / imOr.size  # According to MSE's formula
        MSE_lst.append(MSE)

        if isRGB:
            _newImg=_newImg / 255.0
            imgYIQ[:, :, 0] = _newImg
            _newImg = transformYIQ2RGB(imgYIQ)  # Convert back to RGB
        quantized_lst.append(_newImg)  # add to quantized_lst

        _Z,_Q = fix_boundary(_Z, _Q)  # each boundary become to be a middle of 2 means
        if len(MSE_lst) >= 2:
            if np.abs(MSE_lst[-1] - MSE_lst[-2]) <= 0.000001:
                break

    return quantized_lst, MSE_lst
    pass


def fix_boundary(_Z: np.ndarray, _Q: np.ndarray) -> (List[np.ndarray],List[np.ndarray]):
    for b in range(1, len(_Z) - 1):  # b is boundary
        _Z[b] = (_Q[b - 1] + _Q[b]) / 2
    return _Z, _Q