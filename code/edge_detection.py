import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import filters

img = cv2.imread('../data/HighwayDriving/Train/TrainSeq10/image/TrainSeq10_RGB_Image_0010.png')
'''
LBP for texture abstract
'''
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
radius = 5
no_points = 8 * radius
img_lbp = local_binary_pattern(gray_img, no_points, radius, method='uniform')
img_lbp = np.abs(img_lbp)
img_lbp /= img_lbp.max()
img_lbp = cv2.normalize(img_lbp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite('grad_lbp.png', img_lbp)
print(img_lbp)

cv2.imshow('img', img)
cv2.imshow('lbp', img_lbp)

'''
Sobel cross-gradient operators
'''
grad_sobel = filters.sobel(gray_img)
cv2.imshow('sobel', grad_sobel)
grad_sobel = cv2.normalize(grad_sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite('grad_sobel.png', grad_sobel)

'''
Prewitt cross-gradient operators
'''
grad_prewitt = filters.prewitt(gray_img)
cv2.imshow('prewitt', grad_prewitt)
grad_prewitt = cv2.normalize(grad_prewitt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite('grad_prewitt.png', grad_prewitt)

'''
roberts cross-gradient operators
'''
grad_roberts = filters.roberts(gray_img)
cv2.imshow('roberts', grad_roberts)
grad_roberts = cv2.normalize(grad_roberts, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite('grad_roberts.png', grad_roberts)

'''
Gabor filter
'''
gabor_filter_real, gabor_filter_img = filters.gabor(gray_img, 0.6)
cv2.imshow('gabor_filter_real', gabor_filter_real)
cv2.imshow('gabor_filter_img', gabor_filter_img)
gabor_filter_real = cv2.normalize(gabor_filter_real, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite('gabor_filter_real.png', gabor_filter_real)
cv2.imwrite('gabor_filter_img.png', gabor_filter_img)

'''
Hessian filter
'''
grad_hessian = filters.hessian(img_lbp)
cv2.imshow('grad_hessian', grad_hessian)
grad_hessian = cv2.normalize(grad_hessian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite('grad_hessian.png', grad_hessian)

'''
LoG cross-gradient operators
'''
grad_LoG = filters.laplace(gray_img)
cv2.imshow('LoG', grad_LoG)
grad_LoG = cv2.normalize(grad_LoG, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite('grad_LoG.png', grad_LoG)

cv2.waitKey()