import cv2
import numpy as np

'''
需配置网络摄像机：
1.激活
下载SADP,设置admin密码Cumt123456
2.配置本机IP网卡地址,与网络摄像机在🤝同一网段内
摄像机默认IP地址是:192.168.1.64
设置网卡IP为192.168.1.**均可,
不要与摄像机默认IP地址重复即可
3.读取代码如下
'''

url = 'rtsp://admin:Cumt123456@192.168.1.64//Streaming/Channels/1'
cap = cv2.VideoCapture(url)
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    # 圆形检测https://www.cnblogs.com/wy0904/p/8425447.html
    # 几何形状识别 https://blog.51cto.com/gloomyfish/2104134?lb
    # 图像识别教程 https://blog.csdn.net/feigebabata/article/details.83115056
    result = cv2.blur(frame, (5, 5))  # 降噪处理
    # cv2.imshow('blurframe', result)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)  # 灰度化处理
    # cv2.imshow('gramframe', gray)
    canny = cv2.Canny(frame, 40, 80)  # canny边缘检测
    # cv2.imshow('cannyframe', canny)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
