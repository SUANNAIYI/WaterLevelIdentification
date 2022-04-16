import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab


def cleanlines(lines):
    # 清除重复的线条

    """for lineindex, line in enumerate(lines):
        if line[0]<0:
            lines[lineindex][0] = -line[0]  # rho
            lines[lineindex][1] = line[1]-np.pi  # theta
    newlines = lines
    l_lines = len(lines)
    i = 0
    while i < l_lines:
        flag = 0
        j = i+1
        while j < l_lines:
            flag = 0
            # rho和theta值比较相似或者在一定的范围内的线条归结为一条线(5度以内)
            if (abs(lines[i][1]-lines[j][1])<0.10):
                flag = 1
                lines.pop(j)
                l_lines -= 1
                #print('lines[i]={},lines[j]={},flag={}'.format(lines[i],lines[j],flag))
            j += 1
        newlines = lines
        i += 1
    return newlines"""
    for lineindex, line in enumerate(lines):
        if line[0] < 0:
            lines[lineindex][0] = -line[0]
            lines[lineindex][1] = line[1] - np.pi
    newlines = []
    #newlines.append(lines.pop(6))
    for line in lines:
        flag = 0
        for newline in newlines:
            if ((abs(line[0] - newline[0]) < 10) & (abs(line[1] - newline[1]) < 0.2)):
                flag = 1
        if (flag == 0):
            newlines.append(line)
    return newlines


def IntersectionPoints(lines):
    #求出交点
    points = []
    horLine = []    # 横线
    verLine = []    # 竖线
    horLine_max = []

    for line in lines:
        if((line[1]>(0-0.1))&(line[1]<(0+0.1))):
            horLine_max.append(line)
        else:   #if((line[1] < (np.pi / 4.)) or (line[1] > (3. * np.pi / 4.0))):
            verLine.append(line)


    a = np.array(horLine_max)
    a = a[np.lexsort(-a.T)]
    horLine = [a[0]]    # 取rho最大的horline

    lines_temp = verLine
    newlines = []
    if len(lines_temp) > 3:  # 多于三条直线的，取 rho 值最大的三条
        for i in range(3):
            newlines.append(lines_temp[np.argmax(lines_temp, axis=0)[0]])
            lines_temp.pop(np.argmax(lines_temp, axis=0)[0])  # 删除 rho 最大的直线
        verLine  = newlines
    else:
        verLine = lines_temp

    print('horLine_max=', horLine_max)
    print('horLine=', horLine)
    print('verLine=', verLine)
    for l1 in horLine:
        for l2 in verLine:
            a = np.array([
                [np.cos(l1[1]), np.sin(l1[1])],
                [np.cos(l2[1]), np.sin(l2[1])]
            ])
            b = np.array([l1[0],l2[0]])
            points.append(np.linalg.solve(a, b))
        return points
    #else:
        #print("the number of lines error")



img = cv2.imread("water4.jpg")
width = 400
high = int(img.shape[0]*width/img.shape[1])
image = cv2.resize(img, (width, high))
imgBlur = cv2.GaussianBlur(image, (9, 9), 0)
gray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)   #灰度化处理
Blur =gray.astype("uint8")

x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)       # 转回uint8
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

ret,dst2 = cv2.threshold(absX,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("dst2",dst2)

ret,dst3 = cv2.threshold(Blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("dst3",dst3)

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 14))  # 腐蚀矩阵
iFushi = cv2.morphologyEx(dst3, cv2.MORPH_DILATE, kernel1)  # 对文字腐蚀运算
cv2.imshow("iFushi",iFushi)


kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 30))  # 膨胀矩阵
iPengzhang = cv2.morphologyEx(iFushi, cv2.MORPH_ERODE, kernel2)  # 对背景进行膨胀运算
cv2.imshow("iPengzhang",iPengzhang)

'''# 背景图和二分图相减-->得到文字
jian = np.abs(iPengzhang - dst2)
#cv2.imshow("jian", jian)
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))  # 膨胀
iWenzi = cv2.morphologyEx(jian, cv2.MORPH_DILATE, kernel3)  # 对文字进行膨胀运算
#cv2.imshow('wenzi', iWenzi)'''

edges = cv2.Canny(iPengzhang, 50, 100, apertureSize=3)
cv2.imshow("edges_after", edges)

minLineLength = 10
maxLineGap = 5
# ρ表示从原点到直线的垂直距离，θ表示直线的垂线与横轴顺时针方向的夹角
lines = cv2.HoughLines(edges, 1, np.pi/180, 20)  # 返回（ρ，θ）,ρ的单位是像素，θ是弧度，三轴矩阵
lines = [line[0] for line in lines.tolist()]  # 转成 list
lines = cleanlines(lines)
points = IntersectionPoints(lines)  # 找到交点，求平均值
print('points=',points)

for line in lines:
     rho, theta = line
     print("rho,theta=", rho, theta)
     a = np.cos(theta)
     b = np.sin(theta)
     x0 = a*rho
     y0 = b*rho
     x1 = int(x0 + 2000*(-b))
     y1 = int(y0 + 2000*(a))
     x2 = int(x0 - 2000*(-b))
     y2 = int(y0 - 2000*(a))
     #print((x1,y1),(x2,y2))
     cv2.line(edges,(x1, y1), (x2, y2), (255, 0, 0), 1) #green，画出霍夫变换检测出来的直线

'''for line in lines:
    rho, theta = line
    print("rho,theta=", rho, theta)
    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
        pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
        # 该直线与最后一行的焦点
        pt2 = (int((rho - edges.shape[0] * np.sin(theta)) / np.cos(theta)), edges.shape[0])
        cv2.line(edges, pt1, pt2, (255), 2)  # 绘制一条白线
    else:  # 水平直线
        pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
        # 该直线与最后一列的交点
        pt2 = (edges.shape[1], int((rho - edges.shape[1] * np.cos(theta)) / np.sin(theta)))
        cv2.line(edges, pt1, pt2, (255), 2)  # 绘制一条直线'''

#for point in points:
#     cv2.circle(edges, (int(point[0]), int(point[1])), 5, (255,0,0)) #blue


#array1 = np.array(points)
#array1 = np.sort(array1,axis=0)[::-1]       # 每一列从小到大排序，转置矩阵变从大到小
#print(array1)'''

#max2y = 0
#for j in range(2):
#  max2y+=array1[j][1]       # 第一行的列值+第二行的列值


midx = np.mean([point[0] for point in points])
midy = np.mean([point[1] for point in points])
print('midx={},midy={}'.format(midx, midy))
cv2.circle(edges, (int(midx), int(midy)), 3, (255,0,0))  # blue

plt.figure()
plt.imshow(edges)
plt.show()