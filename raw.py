import cv2
from matplotlib import pyplot as plt
import os
import numpy as np


def cv_show(name, img):
    #
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# plt显示彩色图片
def plt_show0(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()


# plt显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def gray_guss(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image


def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


###水面识别
def cleanlines(lines):
    # 清除重复的线条
    for lineindex, line in enumerate(lines):
        if line[0] < 0:
            lines[lineindex][0] = -line[0]
            lines[lineindex][1] = line[1] - np.pi
    newlines = []
    # newlines.append(lines.pop(6))
    for line in lines:
        flag = 0
        for newline in newlines:
            if ((abs(line[0] - newline[0]) < 10) & (abs(line[1] - newline[1]) < 0.2)):
                flag = 1
        if (flag == 0):
            newlines.append(line)
    return newlines


def IntersectionPoints(lines):
    # 求出交点
    points = []
    horLine = []  # 横线
    verLine = []  # 竖线
    horLine_max = []

    for line in lines:
        if ((line[1] > (0 - 0.1)) & (line[1] < (0 + 0.1))):
            horLine_max.append(line)
        else:  # if((line[1] < (np.pi / 4.)) or (line[1] > (3. * np.pi / 4.0))):
            verLine.append(line)
    a = np.array(horLine_max)
    a = a[np.lexsort(-a.T)]
    horLine = [a[0]]  # 取rho最大的horline
    lines_temp = verLine
    newlines = []
    if len(lines_temp) > 3:  # 多于三条直线的，取 rho 值最大的三条
        for i in range(3):
            newlines.append(lines_temp[np.argmax(lines_temp, axis=0)[0]])
            lines_temp.pop(np.argmax(lines_temp, axis=0)[0])  # 删除 rho 最大的直线
        verLine = newlines
    else:
        verLine = lines_temp

    #    print('horLine_max=', horLine_max)
    #    print('horLine=', horLine)
    #    print('verLine=', verLine)
    for l1 in horLine:
        for l2 in verLine:
            a = np.array([
                [np.cos(l1[1]), np.sin(l1[1])],
                [np.cos(l2[1]), np.sin(l2[1])]
            ])
            b = np.array([l1[0], l2[0]])
            points.append(np.linalg.solve(a, b))
        return points


img = cv2.imread('water4.jpg')
width = 400
high = int(img.shape[0] * width / img.shape[1])
image = cv2.resize(img, (width, high))
imgBlur = cv2.GaussianBlur(image, (9, 9), 0)
gray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)  # 灰度化处理
Blur = gray.astype("uint8")

x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)  # 转回uint8
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

ret, dst2 = cv2.threshold(absX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow("dst2",dst2)

ret, dst3 = cv2.threshold(Blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 14))  # 腐蚀矩阵
iFushi = cv2.morphologyEx(dst3, cv2.MORPH_DILATE, kernel1)  # 对文字腐蚀运算
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 30))  # 膨胀矩阵
iPengzhang = cv2.morphologyEx(iFushi, cv2.MORPH_ERODE, kernel2)  # 对背景进行膨胀运算
edges = cv2.Canny(iPengzhang, 50, 100, apertureSize=3)
minLineLength = 10
maxLineGap = 5
# ρ表示从原点到直线的垂直距离，θ表示直线的垂线与横轴顺时针方向的夹角
lines = cv2.HoughLines(edges, 1, np.pi / 180, 20)  # 返回（ρ，θ）,ρ的单位是像素，θ是弧度，三轴矩阵
lines = [line[0] for line in lines.tolist()]  # 转成 list
lines = cleanlines(lines)
points = IntersectionPoints(lines)  # 找到交点，求平均值
# print('points=',points)

for line in lines:
    rho, theta = line
    # print("rho,theta=", rho, theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))
    # print((x1,y1),(x2,y2))
    cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 1)  # green，画出霍夫变换检测出来的直线

midx = np.mean([point[0] for point in points])
midy = np.mean([point[1] for point in points])
print('midx={},midy={}'.format(midx, midy))
cv2.circle(edges, (int(midx), int(midy)), 3, (255,0,0))
###


origin_image = cv2.imread('water4.jpg', 1)
height, width, channel = origin_image.shape
water = int(midy * height / high)
# print(water)
image1 = origin_image[0:height, 660:720]
# plt_show0(image1)
# 灰度处理
gray_image = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

# 自适应阈值处理
ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
# plt_show(image)

contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 水面分割
imgInfo1 = image1.shape
height, width, channel = image1.shape
# ruler_dst = image1[int(midy*height/high)-200:int(midy*height/high), 0:width]
ruler_dst = image1[int(water - 200):int(water), 0:width]
plt_show0(ruler_dst)

# 水尺图像二值化处理
image3 = cv2.GaussianBlur(ruler_dst, (3, 3), 0)
gray_image = cv2.cvtColor(image3, cv2.COLOR_RGB2GRAY)
# 线性灰度变换
ret, image4 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
plt_show(image4)

# 黑底白字转为白底黑字
area_white = 0
area_black = 0
height, width = image4.shape
for i in range(height):
    for j in range(width):
        if image4[i, j] == 255:
            area_white += 1
        else:
            area_black += 1
if area_white > area_black:
    ret, image4 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    plt_show(image4)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
image5 = cv2.erode(image4, kernel)

# 数字轮廓检测
contours, hierarchy = cv2.findContours(image5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image6 = ruler_dst.copy()
cv2.drawContours(image6, contours, -1, (0, 255, 0), 1)
# plt_show0(image6)

# 数字提取
words = []
for item in contours:
    # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
    word = []
    word_images = []
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    word.append(x)
    word.append(y)
    word.append(weight)
    word.append(height)
    words.append(word)

words = sorted(words, key=lambda s: s[0], reverse=False)
i = 0
for word in words:
    if (word[3] > (word[2] * 3)) and (word[3] < (word[2] * 10)):
        i = i + 1
        image = image6[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
        word_images.append(image)

# 单独显示数字图片
#        plt_show(image)

for i, j in enumerate(word_images):
    plt.subplot(1, 2, i + 1)
    plt.imshow(word_images[i], cmap='gray')
# plt.show()


# 模板
template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# 读取一个文件夹下的所有图片，输入参数是文件名
def read_directory(directory_name):
    referImg_list = []
    for filename in os.listdir(directory_name):
        referImg_list.append(directory_name + "/" + filename)
    return referImg_list


for word_image in word_images:
    image = cv2.GaussianBlur(word_image, (3, 3), 0)
    gray_image = cv2.cvtColor(word_image, cv2.COLOR_RGB2GRAY)
    #    plt_show(gray_image)
    ret, image_ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    #    plt_show(image_)
    c_words = []
    for i in range(0, 10):
        c_word = read_directory('refer10\\' + template[i])
        c_words.append(c_word)
    best_score = []
    for c_word in c_words:
        score = []
        for word in c_word:
            template_img = cv2.imdecode(np.fromfile(word, dtype=np.uint8), 1)
            template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
            ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
            height, width = template_img.shape
            image = image_.copy()
            image = cv2.resize(image, (width, height))
            result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF)
            score.append(result[0][0])
        best_score.append(max(score))
    print(template[0 + best_score.index(max(best_score))], end='')
print('0cm')
