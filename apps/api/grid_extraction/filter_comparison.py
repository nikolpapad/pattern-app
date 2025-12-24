"""
filter_comparison for process.py

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path =r"c:\Users\nikol\Downloads\page_93.png"
img = cv2.imread(img_path)

# Gray scale + light blur to keep edges 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.medianBlur(gray, 3)

#Edges detection
edges = cv2.Canny(gray_blur, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Binarize 
# adaptive threshold (good when lighting varies)
adapt_gaussian_thressholding = cv2.adaptiveThreshold(
    gray_blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,  # invert: lines/filled squares = 1 (white), background = 0 (black)
    15, 10
)

adapt_mean_thressholding = cv2.adaptiveThreshold(
    gray_blur,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,  # invert: lines/filled squares = 1 (white), background = 0 (black)
    15, 10
)

otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

titles = ['Blurred Image', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'Otsu Thresholding']
images = [gray_blur,adapt_mean_thressholding ,adapt_gaussian_thressholding, otsu]
 
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()