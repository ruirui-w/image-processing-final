import cv2
import numpy as np

rgb1 = cv2.imread('12.jpg')
rgb2 = cv2.imread('12.jpg')
rgb3 = cv2.imread('12.jpg')

img = cv2.imread('12.jpg',0)
ret,binary = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
contours,layer_num = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#面积1
area1 = cv2.contourArea(contours[0])
#面积2
_,labels,stats,centroids = cv2.connectedComponentsWithStats(binary)
area2 = stats[1][4]

#周长
length = cv2.arcLength(contours[0],True)

#得到外接矩形的左上角点坐标和长宽
x,y,w,h = cv2.boundingRect(contours[0])

#外接矩形
rect = cv2.rectangle(rgb1,(stats[1][0],stats[1][1]),(stats[1][0]+stats[1][2],stats[1][1]+stats[1][3]),(0,0,255),2)
rect_area = w*h

#最小外接矩形
min_rect = cv2.minAreaRect(contours[0])#返回的是一个元组，第一个元素是左上角点坐标组成的元组，第二个元素是矩形宽高组成的元组，第三个是旋转的角度
# print(min_rect)((530.0443725585938, 113.73445892333984), (40.497230529785156, 137.21890258789062), -86.68222045898438)

#获得最小外接矩形的四个角点坐标
box = cv2.boxPoints(min_rect)#返回的是一个numpy矩阵
min_rect_area = cv2.contourArea(box)

#绘制最小外接矩形，通过多边形绘制的方法进行绘制
box = np.int0(box)#将其转换为整数，否则会报错
min_rect_img = cv2.polylines(rgb2,[box],True,(0,255,0),2)

#细长度
min_rect_h = min_rect[1][0]
min_rect_w = min_rect[1][1]
e = min_rect_h / min_rect_w

#区域占空比（轮廓区域面积除以最小外接矩形面积）
ee = area1 / min_rect_area

#质心
centroid = centroids[0]

text1 = 'height:'+ str(int(min_rect[1][0]))
text2 = "width:" + str(int(min_rect[1][1]))
cv2.putText(rgb1,text1, (10, 30), 3, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
cv2.putText(rgb1,text2, (10, 60), 3, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
cv2.imshow('rect',rect)
cv2.imshow('min_rect',min_rect_img)
cv2.imshow('rgb',rgb3)
cv2.waitKey(0)

print(area1)
print(area2)
print(rect_area)
print(length)
print(min_rect_area)
print(e)
print(ee)
print(centroid)
