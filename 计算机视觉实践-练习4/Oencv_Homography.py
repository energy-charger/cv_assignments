import cv2
import numpy as np

im1 = cv2.imread('pict1.png')
im2 = cv2.imread('pict1.png')

src_points = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
dst_points = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])

H, _ = cv2.findHomography(src_points, dst_points)

h, w = im2.shape[:2]

im2_warp = cv2.warpPerspective(im2, H, (w, h))

cv2.imshow("Warped Source Image", im2_warp)

cv2.waitKey(0)