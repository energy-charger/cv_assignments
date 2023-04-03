import cv2
import numpy as np


def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):
    # 建立暴力匹配器
    matcher = cv2.BFMatcher()
    # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # 存储两个点在featuresA, featuresB中的索引值
            matches.append((m[0].trainIdx, m[0].queryIdx))
        # 当筛选后的匹配对大于4时，计算视角变换矩阵
    if len(matches) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        # 计算视角变换矩阵
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        # 返回结果
        return (matches, H, status)
    # 如果匹配对小于4时，返回None
    return None


# 加载图片
img1 = cv2.imread('imageA.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('imageB.png')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
image1 = gray1.copy()
image2 = gray2.copy()

# 建立SIFT生成器
descriptor = cv2.SIFT_create()
# 检测SIFT特征点，并计算描述子
kps1, features1 = descriptor.detectAndCompute(image1, None)
kps2, features2 = descriptor.detectAndCompute(image2, None)
# 将结果转换成NumPy数组
kps1 = np.float32([kp.pt for kp in kps1])
kps2 = np.float32([kp.pt for kp in kps2])
# 匹配
(matches, H, status) = matchKeypoints(kps1, kps2, features1, features2)
print(H)

if matches is not None:
    # 否则，提取匹配结果
    # 将图片A进行视角变换，result是变换后图片
    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 将图片B传入result图片最左端
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

