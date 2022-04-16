import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import modbus_tk
import modbus_tk.defines as cst
import modbus_tk.modbus_tcp as modbus_tcp
basepath = os.path.dirname(__file__)

# plt显示原图
def cv_show(name, raw_image):
    cv2.imshow(name, raw_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


# plt显示彩色图片
def plt_show0(raw_image):
    blue, green, red = cv2.split(raw_image)
    raw_image = cv2.merge([red, green, blue])
    plt.imshow(raw_image)
    plt.show()


# plt显示灰度图片
def plt_show(raw_image):
    plt.imshow(raw_image, cmap='gray')
    plt.show()


# 高斯滤波
def gray_guss(raw_image):
    rawimage = cv2.GaussianBlur(raw_image, (3, 3), 0)
    grayimage = cv2.cvtColor(rawimage, cv2.COLOR_RGB2GRAY)
    return grayimage


# 清除重复的线条
def cleanLines(raw_lines):
    for lineIndex, line in enumerate(raw_lines):
        if line[0] < 0:
            lines[lineIndex][0] = -line[0]
            lines[lineIndex][1] = line[1] - np.pi
    newlines = []
    # newlines.append(lines.pop(6))
    for line in raw_lines:
        flag = 0
        for newline in newlines:
            if (abs(line[0] - newline[0]) < 10) & (abs(line[1] - newline[1]) < 0.2):
                flag = 1
        if flag == 0:
            newlines.append(line)
    return newlines


# 求出交点
def IntersectionPoints(raw_lines):
    points = []
    # horLine = []  # 横线
    verLine = []  # 竖线
    horLine_max = []

    for line in raw_lines:
        if (line[1] > (0 - 0.1)) & (line[1] < (0 + 0.1)):
            horLine_max.append(line)
        else:
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

    for l1 in horLine:
        for l2 in verLine:
            a = np.array([
                [np.cos(l1[1]), np.sin(l1[1])],
                [np.cos(l2[1]), np.sin(l2[1])]
            ])
            b = np.array([l1[0], l2[0]])
            points.append(np.linalg.solve(a, b))
        return points


rawImage = cv2.imread(basepath+'\\water\\water(456).jpg')  # 读取图片
# 缩放图片，提高识别成功率
width = 400
reducedHeight = int(rawImage.shape[0] * width / rawImage.shape[1])
resizeImage = cv2.resize(rawImage, (width, reducedHeight))
imgBlur = cv2.GaussianBlur(resizeImage, (9, 9), 0)
gray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)  # 灰度化处理
Blur = gray.astype("uint8")

# 边缘检测
x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)  # 转回uint8
absY = cv2.convertScaleAbs(y)

dstImage = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

ret, dst2 = cv2.threshold(absX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv_show("dst2", dst2)


ret3, dst3 = cv2.threshold(Blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 14))  # 腐蚀矩阵
corrosion = cv2.morphologyEx(dst3, cv2.MORPH_DILATE, kernel1)  # 对文字腐蚀运算
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 30))  # 膨胀矩阵
expansion = cv2.morphologyEx(corrosion, cv2.MORPH_ERODE, kernel2)  # 对背景进行膨胀运算
edges = cv2.Canny(expansion, 50, 100, apertureSize=3)
minLineLength = 10
maxLineGap = 5
# ρ表示从原点到直线的垂直距离，θ表示直线的垂线与横轴顺时针方向的夹角
lines = cv2.HoughLines(edges, 1, np.pi / 180, 20)  # 返回（ρ，θ）,ρ的单位是像素，θ是弧度，三轴矩阵
lines = [line[0] for line in lines.tolist()]  # 转成 list
lines = cleanLines(lines)
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
    y1 = int(y0 + 2000 * a)
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * a)
    # print((x1,y1),(x2,y2))
    cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 1)  # green，画出霍夫变换检测出来的直线

midX = np.mean([point[0] for point in points])
midY = np.mean([point[1] for point in points])
print('midX={},midY={}'.format(midX, midY))
cv2.circle(edges, (int(midX), int(midY)), 3, (255, 0, 0))
surfaceHeight = int(midY * rawImage.shape[0] / reducedHeight)  # 放大水面坐标，得到真实像素值
print(surfaceHeight)  # 输出水面高度像素值

# 裁剪图像
rulerImage = rawImage.copy()
plt_show0(rulerImage)
gray_image = cv2.cvtColor(rulerImage, cv2.COLOR_RGB2GRAY)  # 灰度处理
ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)  # 自适应阈值处理
# plt_show(image)

# contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 绘制轮廓

# 水面分割
ruler_dst = rulerImage[int(surfaceHeight - 350):int(surfaceHeight + 100), 0:rawImage.shape[1]]
# plt_show0(ruler_dst)

# 水尺图像二值化处理
numImage = cv2.GaussianBlur(ruler_dst, (3, 3), 0)
gray_image = cv2.cvtColor(numImage, cv2.COLOR_RGB2GRAY)
# 线性灰度变换
ret_numImage, gray_numImage = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
# plt_show(image4)

# 黑底白字转为白底黑字
area_white = 0
area_black = 0
height, width = gray_numImage.shape
for i in range(height):
    for j in range(width):
        if gray_numImage[i, j] == 255:
            area_white += 1
        else:
            area_black += 1
if area_white > area_black:
    ret, gray_numImage = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#    plt_show(image4)

# 7 6
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 6))
number = cv2.erode(gray_numImage, kernel)
dilateImage = cv2.dilate(number, kernel)

# 数字轮廓检测
contours, hierarchy = cv2.findContours(dilateImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# image6 = ruler_dst.copy()
cv2.drawContours(ruler_dst, contours, -1, (0, 0, 255), 1)
plt_show0(ruler_dst)

# 数字提取
words = []
word_images = []
for item in contours:
    word = []
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if rect[2] > 12 and rect[3] > 60:
        word.append(x)
        word.append(y)
        word.append(weight)
        word.append(height)
        words.append(word)

words = sorted(words, key=lambda s: s[0], reverse=False)
i = 0

# 筛选边缘检测出的数字
for word in words:
    if (word[3] > (word[2] * 2.5)) and (word[3] < (word[2] * 10)):
        i = i + 1
        image = ruler_dst[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
        word_images.append(image)

#  判断图片远近情况
if len(word_images) == 2:
    for word_image in word_images:
        plt_show(word_image)

#   未检测出两个数字
else:
    ruler_dst = rulerImage[int(surfaceHeight - 190):int(surfaceHeight + 30), 0:rawImage.shape[1]]
    numImage = cv2.GaussianBlur(ruler_dst, (1, 1), 0)  # 水尺图像二值化处理
    gray_image = cv2.cvtColor(numImage, cv2.COLOR_RGB2GRAY)
    ret_numImage, gray_numImage = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)  # 线性灰度变换
    # 黑底白字转为白底黑字
    area_white = 0
    area_black = 0
    height, width = gray_numImage.shape
    for i in range(height):
        for j in range(width):
            if gray_numImage[i, j] == 255:
                area_white += 1
            else:
                area_black += 1
    if area_white > area_black:
        ret, gray_numImage = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    number = cv2.erode(gray_numImage, kernel)
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilateImage = cv2.dilate(number, kernel)

    # 数字轮廓检测
    contours, hierarchy = cv2.findContours(dilateImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # image6 = ruler_dst.copy()
    cv2.drawContours(ruler_dst, contours, -1, (0, 0, 255), 1)
    plt_show0(ruler_dst)
    # 数字提取
    words = []
    word_images = []
    for item in contours:
        word = []
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        if rect[2] > 8 and rect[3] > 40:
            word.append(x)
            word.append(y)
            word.append(weight)
            word.append(height)
            words.append(word)
    words = sorted(words, key=lambda s: s[0], reverse=False)
    i = 0
    # 筛选边缘检测出的数字
    for word in words:
        if (word[3] > (word[2] * 2.5)) and (word[3] < (word[2] * 10)):
            i = i + 1
            image = ruler_dst[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            word_images.append(image)
            plt_show(image)

# 模板
template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# 读取一个文件夹下的所有图片，输入参数是文件名
def read_directory(directory_name):
    referimg_list = []
    for filename in os.listdir(directory_name):
        referimg_list.append(directory_name + "/" + filename)
    return referimg_list


hundred = ten = 0
j = 0

for word_image in word_images:
    image = cv2.GaussianBlur(word_image, (3, 3), 0)  # 高斯滤波
    gray_image = cv2.cvtColor(word_image, cv2.COLOR_RGB2GRAY)
    #    plt_show(gray_image)
    ret, singleNum = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    #    plt_show(SingleNum)
    c_words = []
    for i in range(0, 10):
        c_word = read_directory(basepath+'\\refer10\\' + template[i])
        c_words.append(c_word)
    best_score = []
    for c_word in c_words:
        score = []
        for word in c_word:
            template_img = cv2.imdecode(np.fromfile(word, dtype=np.uint8), 1)
            template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
            ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
            height, width = template_img.shape
            image = singleNum.copy()
            image = cv2.resize(image, (width, height))
            result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF)  # 模板匹配，计算得分
            score.append(result[0][0])
        best_score.append(max(score))
    if j == 0:
        hundred = best_score.index(max(best_score))  # 获取百位数
        j = j + 1
    else:
        ten = best_score.index(max(best_score))  # 获取十位数
num = 100*hundred+10*ten  # 获取水位高度
print(num, "cm")

logger = modbus_tk.utils.create_logger("console")
if __name__ == "__main__":
    try:
        # 连接MODBUS TCP从机
        master = modbus_tcp.TcpMaster(host="10.4.60.120")  # ip为从机ip
        master.set_timeout(5.0)
        logger.info("connected")
        logger.info(master.execute(1, cst.WRITE_SINGLE_REGISTER, 0, output_value=num))
        # 写入从站保持寄存器：0代表写入寄存器的地址，output_value为写入的数值
        #logger.info(master.execute(1, cst.READ_HOLDING_REGISTERS, 0, 1))  # 读从站保持寄存器
    except modbus_tk.modbus.ModbusError as e:
        logger.error("%s- Code=%d" % (e, e.get_exception_code()))
