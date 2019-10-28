import cv2
import numpy as np

img = cv2.imread('lena.jpg')
res = np.uint8(np.clip((1.5 * img + 10), 0, 255))
tmp = np.hstack((img, res))  # 两张图片横向合并（便于对比显示）

cv2.imshow('image', tmp)
cv2.imwrite('lenaByLena.jpg',tmp)
cv2.waitKey(0)