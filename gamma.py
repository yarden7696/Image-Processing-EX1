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
import numpy as np
import cv2
from ex1_utils import LOAD_GRAY_SCALE


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    if (rep == 2):  # RGB
        img = cv2.imread(img_path)
    else:  # GrayScale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    temp = img
    cv2.imshow('Gamma Correction', img)

    # I create an inner function that her input is a gamma number. this function displays the
    # new image according to the new gamma.
    def display(position: int):
        position = float(position)
        position = position / 50
        _img = np.uint8(255 * np.power((temp / 255), position))
        cv2.imshow('Gamma Correction', _img)

    cv2.namedWindow('Gamma Correction')
    # Creates a trackbar and attaches it to the specified window.
    cv2.createTrackbar('Gamma', 'Gamma Correction', 1, 100, display)
    display(1)

    while True:
        key = cv2.waitKey(1000)
        if key == 27:
            break
        if cv2.getWindowProperty("Gamma Correction", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    pass


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()




