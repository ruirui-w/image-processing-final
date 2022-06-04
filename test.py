import cv2
import matplotlib.pyplot as plt
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)  #参数为视频设备的id
#如果只有一个摄像头可以填0，表示打开默认的摄像头,
# 这里的参数也可以是视频文件名路径，只要把视频文件的具体路径写进去就好
while True:
    ret, frame = cap.read()
    if frame.any() != None:
        cv2.imshow('capture',frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.destroyWindow('capture')
        break
cap.release()
cv2.imshow('result',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()