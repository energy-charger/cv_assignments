# -*- coding: utf-8 -*-
import cv2
import numpy as np
# 读取图片
im1 = cv2.imread('pic1.jpg')
im2 = cv2.imread('pic2.jpg')

# 计算SURF特征点和对应的描述子，kp存储特征点坐标，des存储对应描述子
surf = cv2.xfeatures2d.SURF_create()
kp1, des1 = surf.detectAndCompute(im1, None)
kp2, des2 = surf.detectAndCompute(im2, None)

# 匹配特征点描述子
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 提取匹配较好的特征点
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# 通过特征点坐标计算单应性矩阵H
# （findHomography中使用了RANSAC算法剔除错误匹配）
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

# 使用单应性矩阵计算变换结果并绘图
h, w, d = im1.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, H)

img2 = cv2.polylines(im2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)


draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

im3 = cv2.drawMatches(im1, kp1, im2, kp2, good, None, **draw_params)
im2_warp = cv2.warpPerspective(im1, H, (w, h))
cv2.imwrite('result.png',im2_warp)