import cv2
import numpy as np
from scipy import signal, misc
import matplotlib as plt

img = cv2.imread('HighwayDriving/Train/TrainSeq00/image/TrainSeq00_RGB_Image_0000.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_xy = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_yx = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])

roberts_x = np.array([[-1, 0], [0, 1]])
roberts_y = np.array([[0, -1], [1, 0]])

prewitt_x = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
prewitt_xy = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
prewitt_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_yx = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])

LoG = np.array([[0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0]])
'''
Sobel cross-gradient operators
'''
grad_x_sobel = signal.convolve2d(gray_img, sobel_x, boundary='symm', mode='same')
grad_xy_sobel = signal.convolve2d(gray_img, sobel_xy, boundary='symm', mode='same')
grad_y_sobel = signal.convolve2d(gray_img, sobel_y, boundary='symm', mode='same')
grad_yx_sobel = signal.convolve2d(gray_img, sobel_yx, boundary='symm', mode='same')

grad_x_sobel[grad_x_sobel < 0] = 0
grad_xy_sobel[grad_xy_sobel < 0] = 0
grad_y_sobel[grad_y_sobel < 0] = 0
grad_yx_sobel[grad_yx_sobel < 0] = 0
grad_sobel = (grad_x_sobel + grad_xy_sobel + grad_y_sobel + grad_yx_sobel)/4
grad_sobel = grad_sobel.astype('uint8')
cv2.imshow('sobel', grad_sobel)

'''
Prewitt cross-gradient operators
'''
grad_x_prewitt = signal.convolve2d(gray_img, prewitt_x, boundary='symm', mode='same')
grad_xy_prewitt = signal.convolve2d(gray_img, prewitt_xy, boundary='symm', mode='same')
grad_y_prewitt = signal.convolve2d(gray_img, prewitt_y, boundary='symm', mode='same')
grad_yx_prewitt = signal.convolve2d(gray_img, prewitt_yx, boundary='symm', mode='same')

grad_x_prewitt[grad_x_prewitt < 0] = 0
grad_xy_prewitt[grad_xy_prewitt < 0] = 0
grad_y_prewitt[grad_y_prewitt < 0] = 0
grad_yx_prewitt[grad_yx_prewitt < 0] = 0
grad_prewitt = (grad_x_prewitt + grad_xy_prewitt + grad_y_prewitt + grad_yx_prewitt)/4
grad_prewitt = grad_prewitt.astype('uint8')
cv2.imshow('prewitt', grad_prewitt)

'''
roberts cross-gradient operators
'''
grad_x_roberts = signal.convolve2d(gray_img, roberts_x, boundary='symm', mode='same')
grad_y_roberts = signal.convolve2d(gray_img, roberts_y, boundary='symm', mode='same')

grad_x_roberts[grad_x_roberts < 0] = 0
grad_y_roberts[grad_y_roberts < 0] = 0
grad_roberts = (grad_x_roberts + grad_y_roberts)
grad_roberts = grad_roberts.astype('uint8')
cv2.imshow('roberts', grad_roberts)

'''
LoG cross-gradient operators
'''
grad_LoG = signal.convolve2d(gray_img, LoG, boundary='symm', mode='same')
grad_LoG[grad_LoG < 0] = 0
grad_LoG = grad_LoG.astype('uint8')
cv2.imshow('LoG', grad_LoG)

cv2.waitKey()