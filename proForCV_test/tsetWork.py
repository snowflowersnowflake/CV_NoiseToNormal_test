import cv2
import numpy as np
import random

img_path='1024.png'
img=cv2.imread(img_path)

#makegary
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#cv2.imshow("img_gray",img_gray)
#cv2.imwrite('.1024gray.png',img_gray)

#makeEqual
img2_gray=cv2.equalizeHist(img_gray)
#cv2.imshow("img_gray",img2_gray)
#cv2.imwrite('.1024gray_equalizeHist.png',img2_gray)

#makenoise
def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 cv2.imshow('x',img)
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

spNoiseImg=sp_noise(img,0.1)
#cv2.imshow('noise',spNoiseImg)
#cv2.imwrite('spNoiseImg.png',spNoiseImg)

def gasuss_noise(image, var,mean=0 ):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out


gasNoiseImg=gasuss_noise(img,0.05)
#cv2.imshow('noise',gasNoiseImg)
#cv2.imwrite('gasNoiseImg.png',gasNoiseImg)


#meanfilter
dst1 = cv2.blur(spNoiseImg, (3, 3))
dst2 = cv2.blur(spNoiseImg, (5, 5))
dst3 = cv2.blur(spNoiseImg, (7, 7))

dst4 = cv2.blur(gasNoiseImg,(3, 3))
dst5 = cv2.blur(gasNoiseImg,(5, 5))
dst6 = cv2.blur(gasNoiseImg,(7, 7))

dst7 = cv2.medianBlur(spNoiseImg, 3)
dst8 = cv2.medianBlur(spNoiseImg, 5)
dst9 = cv2.medianBlur(spNoiseImg, 7)

dst10 = cv2.medianBlur(gasNoiseImg,3)
dst11 = cv2.medianBlur(gasNoiseImg,5)
dst12 = cv2.medianBlur(gasNoiseImg,7)

cv2.imshow('dst',dst12)
cv2.imwrite('7mediGa.png',dst12)

cv2.waitKey(0)

#medifilter




























