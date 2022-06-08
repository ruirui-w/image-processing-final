import math
import os
import random
import sys
import winsound

from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import cv2
import numpy
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QGraphicsScene, QFileDialog, QMessageBox
from PySimpleGUI import PySimpleGUI
from matplotlib import pyplot as plt

from mainWindow import Ui_MainWindow
from propertyWindow import Ui_Form
import face_recognition
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import QWidget, QApplication,\
    QColorDialog, QFrame, QPushButton
from PyQt5.QtGui import QColor
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Leye_cascade = cv2.CascadeClassifier('aarcascade_lefteye_2splits.xml')
Reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
simulate_real_time = "true"
process_eye = 0
eyeq_len = 5
eyeq = []
# 预处理窗口类
# 就是弹出来的调整各种属性值的小窗口
class PropertyWindow(QWidget, Ui_Form):
    # 信号只能在Object的子类中创建，并且只能在创建类的时候的时候添加，而不能等类已经定义完再作为动态属性添加进去。
    # 自定义的信号在__init__()函数之前定义。
    # 自定义一个信号signal，有一个object类型的参数
    # str->bug
    signal = QtCore.pyqtSignal(object)

    # 类初始化
    def __init__(self):
        # 调用父类的初始化
        super(PropertyWindow, self).__init__()
        # 窗口界面初始化
        self.setupUi(self)

        # 绑定窗口组件响应事件的处理函数（将窗口中的组件被用户触发的点击、值变化等事件绑定到处理函数）
        # 数值框的值改变
        self.spinBox.valueChanged.connect(self.__spinBoxChange)
        # 滑动条的值改变
        self.slider.valueChanged.connect(self.__sliderChange)
        # 点击确认按钮
        self.submitButton.clicked.connect(self.__valueConfirm)

    # 数值框值改变的处理函数
    def __spinBoxChange(self):
        # 获取数值框的当前值
        value = self.spinBox.value()
        # 与滑动条进行数值同步
        self.slider.setValue(value)
        # 发送信号到主窗口，参数是当前数值（主窗口有自定义的接收并处理该信号的函数）
        self.signal.emit(value)

    # 滑动条值改变的处理函数
    def __sliderChange(self):
        # 获取滑动条的当前值
        value = self.slider.value()
        # 与数值框进行数值同步
        # 注意：该操作也会触发数值框的数值改变，即会触发调用__spinBoxChange()，所以不要需要在此处重复发信号到主窗口
        self.spinBox.setValue(value)

    # 确认按钮按下的处理函数
    def __valueConfirm(self):
        # 发送确认修改信号
        self.signal.emit('ok')
        # 关闭窗口
        self.close()

    # 重写窗口关闭处理
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        # 发送取消修改信号，关闭窗口时触发
        self.signal.emit('close')


# 主窗口类
class MainWindow(QMainWindow, Ui_MainWindow):
    # 类初始化
    def __init__(self):
        # 提示窗口标识初始化
        self.tezhengflag = 0
        # 调用父类的初始化
        super(MainWindow, self).__init__()
        # 窗口界面初始化
        self.setupUi(self)

        # 图像属性调整的子窗口，初始化默认空（亮度、对比度、锐度、饱和度、色调、旋转、缩放调节窗口）
        self.__propertyWindow = None

        # 当前打开的图片文件名，初始化默认空
        self.__fileName = None

        # 保存图片原始数据，初始化默认空
        self.__srcImageRGB = None
        # 保存图片最终处理结果的数据，初始化默认空
        self.__outImageRGB = None
        # 保存图片暂时修改的数据，初始化默认空
        # （在修改图像属性未点击确认时，需要暂存修改数据，如果确认后将临时数据同步为最终结果数据，如果未确认将复原数据）
        self.__tempImageRGB = None

        # 绑定窗口事件的响应函数
        # 文件菜单
        # 打开文件
        self.openFileAction.triggered.connect(self.__openFileAndShowImage)
        # 摄像头捕捉
        self.getFileAction.triggered.connect(self.__getFileAndShowImage)
        # 保存文件
        self.saveFileAction.triggered.connect(self.saveFile)
        # 另存为文件
        self.saveFileAsAction.triggered.connect(self.saveFileAs)
        # 退出程序
        self.exitAppAction.triggered.connect(self.close)

        # 重置图像菜单
        # 重置图像
        self.resetImageAction.triggered.connect(self.__resetImage)
        # 直接灰度映射菜单
        # 灰度化
        self.grayAction.triggered.connect(self.__toGrayImage)
        # 二值化
        self.binaryAction.triggered.connect(self.__toBinaryImage)
        # 颜色反转
        self.reverseAction.triggered.connect(self.__reverseImage)
        # 亮度调整
        self.lightAction.triggered.connect(self.__openLightWindow)
        # 对比度调整
        self.contrastAction.triggered.connect(self.__openContrastWindow)
        # 锐度调整
        self.sharpAction.triggered.connect(self.__openSharpWindow)
        # 饱和度调整
        self.saturationAction.triggered.connect(self.__openSaturationWindow)
        # 色度调整
        self.hueAction.triggered.connect(self.__openHueWindow)

        # 图像运算菜单
        # 加
        self.imageAddAction.triggered.connect(self.__addImage)
        # 减
        self.imageSubtractAction.triggered.connect(self.__subtractImage)
        # 乘
        self.imageMultiplyAction.triggered.connect(self.__multiplyImage)
        # 融合
        self.imagefusionAction.triggered.connect(self.__fusionImage)
        self.imagefusionzixuanAction.triggered.connect(self.__fusionImagezixuan)
        # 缩放
        self.zoomAction.triggered.connect(self.__openZoomWindow)
        # 旋转
        self.rotateAction.triggered.connect(self.__openRotateWindow)
        # 垂直对称
        self.imagechuizhiAction.triggered.connect(self.__chuizhiduicheng)
        # 水平对称
        self.imageshuipingAction.triggered.connect(self.__shuipingduicheng)
        # 镜像对称
        self.imagejingxiangduichengAction.triggered.connect(self.__jingxiangduicheng)
        # 直方图均衡菜单
        # 归一化直方图
        self.histogramAction.triggered.connect(self.__histogram)
        # 直方图均衡化
        self.histogramEqAction.triggered.connect(self.__histogramEqualization)

        # 特征描述菜单
        # 面积
        self.quyumianjiAction.triggered.connect(self.__quyumianji)
        # 周长
        self.quyuzhouchangAction.triggered.connect(self.__quyuzhouchang)
        # 最小外接矩形
        self.quyujuxingAction.triggered.connect(self.__quyujuxing)
        # 矩形度
        self.quyuzhankongbiAction.triggered.connect(self.__quyujuxingdu)
        # 细长度
        self.quyuxichangduAction.triggered.connect(self.__quyuxichangdu)
        # 圆形度
        self.quyuyuanxingduAction.triggered.connect(self.__quyuyuanxingdu)
        #球状度
        self.quyuqiuzhuangduAction.triggered.connect(self.__quyuqiuzhuangdu)
        # 重心
        self.quyuzhongxinAction.triggered.connect(self.__quyuzhongxin)
        # Harris角
        self.quyuHarrisAction.triggered.connect(self.__quyuHarris)
        self.quyusubHarrisAction.triggered.connect(self.__quyusubHarris)
        # 检测圆
        self.quyujianceyuanAction.triggered.connect(self.__quyujianceyuan)
        # 轮廓识别
        self.quyulunkunshibieAction.triggered.connect(self.__quyulunkuoshibie)
        # 图像截取菜单
        # 裁剪
        self.caijianAction.triggered.connect(self.__caijian)
        # 迭代阈值分割
        self.diedaiAction.triggered.connect(self.__diedai)
        # 种子填充
        self.zhongziAction.triggered.connect(self.__zhongzi)
        # 人脸框取
        self.renlianAction.triggered.connect(self.__renlianjiequ)
        # 人脸轮廓(单张脸)
        self.renlianjiequAction.triggered.connect(self.__renlianjiequ2)
        # 噪声菜单
        # 加高斯噪声
        self.addGaussianNoiseAction.triggered.connect(self.__addGasussNoise)
        # 加均匀噪声
        self.addUiformNoiseAction.triggered.connect(self.__addUniformNoise)
        # 加脉冲（椒盐）噪声
        self.addImpulseNoiseAction.triggered.connect(self.__addImpulseNoise)
        # 加随机噪声
        self.addbosongNoiseAction.triggered.connect(self.__addbosongNoise)

        # 空域滤波菜单
        # 均值滤波
        self.meanValueAction.triggered.connect(self.__meanValueFilter)
        # 中值滤波
        self.medianValueAction.triggered.connect(self.__medianValueFilter)
        # 自实现中值滤波
        self.zishixianmedianValueAction.triggered.connect(self.__zishixianmedianValueFilter)
        # 高斯滤波
        self.guasslvboAction.triggered.connect(self.__guasslvbo)
        # Sobel算子锐化
        self.sobelAction.triggered.connect(self.__sobel)
        # Prewitt算子锐化
        self.prewittAction.triggered.connect(self.__prewitt)
        # 拉普拉斯算子锐化
        self.laplacianAction.triggered.connect(self.__laplacian)
        # Canny算子锐化
        self.CannyAction.triggered.connect(self.__CannyAction)
        # 图片美化效果菜单
        # 浮雕效果
        self.fudiaoAction.triggered.connect(self.__fudiao)
        # 毛玻璃效果
        self.maoboliAction.triggered.connect(self.__maoboli)
        # 图像卡通化
        self.katonghuaAction.triggered.connect(self.__katonghua)
        # 马赛克效果
        self.masaikeAction.triggered.connect(self.__masaike)

        # 加框
        self.biankuangAction.triggered.connect(self.__jiakuang)
        # 融合加框
        self.ronghekuangAction.triggered.connect(self.__ronghekuang)
        # 拼图
        self.pingtuAction.triggered.connect(self.__pingtu)
        # 相片滤镜菜单
        # 怀旧
        self.huaijiuAction.triggered.connect(self.__huaijiu)
        # 光晕
        self.guangyunAction.triggered.connect(self.__guangyun)
        # 流年
        self.liunianAction.triggered.connect(self.__liunian)
        # 一键美化
        self.meihuaAction.triggered.connect(self.__renlianmeihua)
        # 人脸一键美化化妆
        self.renlianmakeupAction.triggered.connect(self.__renlianmakeup)
        # 关于菜单
        # 关于作者
        self.aboutAction.triggered.connect(self.__aboutAuthor)
        #学习侦测
        self.xuexizhenceAction.triggered.connect(self.__kaishi)

    # -----------------------------------文件-----------------------------------
    # 打开文件并在主窗口中显示打开的图像
    def __openFileAndShowImage(self):
        # 打开文件选择窗口
        __fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image Files(*.png *.jpeg *.jpg *.bmp)')
        # 文件存在
        if __fileName and os.path.exists(__fileName):
            # 设置打开的文件名属性
            self.__fileName = __fileName
            # 转换颜色空间，cv2默认打开BGR空间，Qt界面显示需要RGB空间，所以就全部统一到RGB显示吧
            __bgrImg = cv2.imread(self.__fileName)
            # 设置初始化数据
            self.__srcImageRGB = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
            self.__outImageRGB = self.__srcImageRGB.copy()
            self.__tempImageRGB = self.__srcImageRGB.copy()
            # 在窗口中左侧QGraphicsView区域显示图片
            self.__drawImage(self.srcImageView, self.__srcImageRGB)
            # 在窗口中右侧QGraphicsView区域显示图片
            self.__drawImage(self.outImageView, self.__srcImageRGB)

    # 在窗口中指定的QGraphicsView区域（左或右）显示指定类型（rgb、灰度、二值）的图像
    def __drawImage(self, location, img):
        # RBG图
        if len(img.shape) > 2:
            # 获取行（高度）、列（宽度）、通道数
            __height, __width, __channel = img.shape
            # 转换为QImage对象，注意第四、五个参数
            __qImg = QImage(img, __width, __height, __width * __channel, QImage.Format_RGB888)
        # 灰度图、二值图
        else:
            # 获取行（高度）、列（宽度）、通道数
            __height, __width = img.shape
            # 转换为QImage对象，注意第四、五个参数
            __qImg = QImage(img, __width, __height, __width, QImage.Format_Indexed8)

        # 创建QPixmap对象
        __qPixmap = QPixmap.fromImage(__qImg)
        # 创建显示容器QGraphicsScene对象
        __scene = QGraphicsScene()
        # 填充QGraphicsScene对象
        __scene.addPixmap(__qPixmap)
        # 将QGraphicsScene对象设置到QGraphicsView区域实现图片显示
        location.setScene(__scene)

    # 执行保存图片文件的操作
    def __saveImg(self, fileName):
        # 已经打开了文件才能保存
        if self.__fileName:
            # RGB转BRG空间后才能通过opencv正确保存
            __bgrImg = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2BGR)
            # 保存
            cv2.imwrite(fileName, __bgrImg)
            # 消息提示窗口
            QMessageBox.information(self, '提示', '文件保存成功！')
        else:
            # 消息提示窗口
            QMessageBox.information(self, '提示', '文件保存失败！')

    # 保存文件，覆盖原始文件
    def saveFile(self):
        __fileName = 'result.jpg'
        self.__saveImg(__fileName)

    # 文件另存
    def saveFileAs(self):
        # 已经打开了文件才能保存
        if self.__fileName:
            try:
                # 打开文件保存的选择窗口
                __fileName, _ = QFileDialog.getSaveFileName(self, '保存图片', 'Image',
                                                            'Image Files(*.png *.jpeg *.jpg *.bmp)')
                self.__saveImg(__fileName)
            except:
                QMessageBox.information(self, '提示', '文件保存失败！')
        else:
            # 消息提示窗口
            QMessageBox.information(self, '提示', '文件保存失败！')

    # 重写窗口关闭事件函数，来关闭所有窗口。因为默认关闭主窗口子窗口依然存在。
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        reply = QMessageBox.question(self, '警告', '确认退出？', QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            sys.exit(0)
        else:
          return
    # 摄像头捕捉事件
    def __getFileAndShowImage(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 参数为视频设备的id
        # 如果只有一个摄像头可以填0，表示打开默认的摄像头,
        # 这里的参数也可以是视频文件名路径，只要把视频文件的具体路径写进去就好
        while True:
            ret, frame = cap.read()
            if frame.any() != None:
                cv2.imshow('capture', frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                cv2.destroyAllWindows()
                break
        cap.release()
        cv2.imwrite('photo.jpg', frame)
        self.__fileName = 'photo.jpg'
        # 转换颜色空间，cv2默认打开BGR空间，Qt界面显示需要RGB空间，所以就统一到RGB吧
        __bgrImg = cv2.imread(self.__fileName)
        # 设置初始化数据
        self.__srcImageRGB = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
        self.__outImageRGB = self.__srcImageRGB.copy()
        self.__tempImageRGB = self.__srcImageRGB.copy()
        # 在窗口中左侧QGraphicsView区域显示图片
        self.__drawImage(self.srcImageView, self.__srcImageRGB)
        # 在窗口中右侧QGraphicsView区域显示图片
        self.__drawImage(self.outImageView, self.__srcImageRGB)
        QMessageBox.information(self, '提示', '相片已经保存至当前目录！')

    # -----------------------------------重置图片-----------------------------------
    # 重置图片到初始状态
    def __resetImage(self):
        if self.__fileName:
            # 还原文件打开时的初始化图片数据
            self.__outImageRGB = self.__srcImageRGB.copy()
            # 窗口显示图片
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # -----------------------------------图像预处理-----------------------------------
    # 灰度化
    def __toGrayImage(self):
        # 只有RGB图才能灰度化
        if self.__fileName and len(self.__outImageRGB.shape) > 2:
            # 灰度化使得三通道RGB图变成单通道灰度图
            self.__outImageRGB = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '目前的图像已经是灰度图了！')

    # 二值化
    def __toBinaryImage(self):
        # 先灰度化
        if self.__fileName:
            # 后阈值化为二值图
            self.__toGrayImage()
            _, self.__outImageRGB = cv2.threshold(self.__outImageRGB, 127, 255, cv2.THRESH_BINARY)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # 反转图片颜色
    def __reverseImage(self):
        if self.__fileName:
            self.__outImageRGB = cv2.bitwise_not(self.__outImageRGB)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # 执行打开属性调节子窗口（亮度、对比度、锐度、饱和度、色调、缩放、旋转）
    def __openPropertyWindow(self, propertyName, func):
        if self.__fileName:
            if self.__propertyWindow:
                self.__propertyWindow.close()
            self.__propertyWindow = PropertyWindow()
            # 设置窗口内容
            self.__propertyWindow.setWindowTitle(propertyName)
            self.__propertyWindow.propertyLabel.setText(propertyName)
            # 接收信号
            # 设置主窗口接收子窗口发送的信号的处理函数
            self.__propertyWindow.signal.connect(func)
            # 禁用主窗口菜单栏，子窗口置顶，且无法切换到主窗口
            self.__propertyWindow.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
            self.__propertyWindow.setWindowModality(QtCore.Qt.ApplicationModal)
            # 显示子窗口
            self.__propertyWindow.show()
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 亮度调节子窗口
    def __openLightWindow(self):
        self.__openPropertyWindow('亮度', self.__changeLight)

    # 对比度调节子窗口
    def __openContrastWindow(self):
        self.__openPropertyWindow('对比度', self.__changeContrast)

    # 锐度调节子窗口
    def __openSharpWindow(self):
        self.__openPropertyWindow('锐度', self.__changeSharp)

    # 饱和度调节子窗口
    def __openSaturationWindow(self):
        self.__openPropertyWindow('饱和度', self.__changeSaturation)

    # 色调调节子窗口
    def __openHueWindow(self):
        self.__openPropertyWindow('色调', self.__changeHue)

    # 预处理信号
    def __dealSignal(self, val):
        # 拷贝后修改副本
        __img = self.__outImageRGB.copy()
        # 如果是灰度图要转为RGB图
        if len(__img.shape) < 3:
            __img = cv2.cvtColor(__img, cv2.COLOR_GRAY2RGB)

        value = str(val)
        # 确认修改
        if value == 'ok':
            # 将暂存的修改保存为结果
            self.__outImageRGB = self.__tempImageRGB.copy()
            return None
        # 修改完成（确认已经做的修改或取消了修改）
        elif value == 'close':
            # 重绘修改预览
            self.__drawImage(self.outImageView, self.__outImageRGB)
            return None
        # 暂时修改
        else:
            return __img

    # 执行改变亮度或对比度
    # g(i,j)=αf(i,j)+(1-α)black+β，α用来调节对比度, β用来调节亮度
    def __lightAndContrast(self, img, alpha, beta):
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        rows, cols, channels = img.shape
        # 新建全零(黑色)图片数组
        blank = numpy.zeros([rows, cols, channels], img.dtype)
        # 计算两个图像阵列的加权和
        img = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
        # 显示修改数据
        self.__drawImage(self.outImageView, img)
        return img

    # 修改亮度
    def __changeLight(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        # None的size是1 ！！！
        if numpy.size(__img) > 1:
            beta = int(val) * (255 / 100)
            # 暂存本次修改
            self.__tempImageRGB = self.__lightAndContrast(__img, 1, beta)

    # 修改对比度
    def __changeContrast(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        if numpy.size(__img) > 1:
            k = int(val)
            if k != -100:
                alpha = (k + 100) / 100
            else:
                alpha = 0.01
            # 暂存本次修改
            self.__tempImageRGB = self.__lightAndContrast(__img, alpha, 0)

    # 修改锐度
    def __changeSharp(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        if numpy.size(__img) > 1:
            # 比例
            k = int(val) * 0.01
            if k != 0:
                # 卷积核（拉普拉斯算子）
                kernel = numpy.array([[-1, -1, -1], [-1, 9 + k, -1], [-1, -1, -1]])
                # 通过卷积实现锐化,暂存修改数据
                self.__tempImageRGB = cv2.filter2D(__img, -1, kernel)
            else:
                self.__tempImageRGB = self.__outImageRGB.copy()
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # 修改饱和度
    def __changeSaturation(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        if numpy.size(__img) > 1:
            # 转换颜色空间到HLS
            __img = cv2.cvtColor(__img, cv2.COLOR_RGB2HLS)
            # 比例
            k = int(val) * (255 / 100)
            # 切片修改S分量，并限制色彩数值在0-255之间
            __img[:, :, 2] = numpy.clip(__img[:, :, 2] + k, 0, 255)
            # 暂存修改数据
            self.__tempImageRGB = cv2.cvtColor(__img, cv2.COLOR_HLS2RGB)
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # 修改色调
    # OpenCV中hue通道的取值范围是0 - 180
    def __changeHue(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        if numpy.size(__img) > 1:
            # 转换颜色空间到HLS
            __img = cv2.cvtColor(__img, cv2.COLOR_RGB2HLS)
            # 比例
            k = int(val) * (90 / 100)
            # 切片修改H分量，并限制色彩数值在0-180之间
            __img[:, :, 0] = (__img[:, :, 0] + k) % 180
            # 暂存修改数据
            self.__tempImageRGB = cv2.cvtColor(__img, cv2.COLOR_HLS2RGB)
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # -----------------------------------图像运算-----------------------------------
    # 加、减、乘、融合操作
    def __operation(self, func):
        if self.__fileName:
            __fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image Files(*.png *.jpeg *.jpg *.bmp)')
            if __fileName and os.path.exists(__fileName):
                __bgrImg = cv2.imread(__fileName)
                # 图片尺寸相同才能进行运算
                if self.__outImageRGB.shape == __bgrImg.shape:
                    # 一定要转颜色空间！！！
                    __rgbImg = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
                    self.__outImageRGB = func(self.__outImageRGB, __rgbImg)
                    self.__drawImage(self.outImageView, self.__outImageRGB)
                else:
                    reply = QMessageBox.question(self, '确认完成操作吗', '提示：图片尺寸不一致，操作可能不好', QMessageBox.Yes,
                                                 QMessageBox.No)
                    if reply == QMessageBox.No:
                        return
                    else:
                        # 一定要转颜色空间！
                        __rgbImg = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
                        h0, w0 = self.__outImageRGB.shape[0], self.__outImageRGB.shape[1]  # cv2 读取出来的是h,w,c
                        h1, w1 = __rgbImg.shape[0], __rgbImg.shape[1]
                        h = max(h0, h1)
                        w = max(w0, w1)
                        org_image = numpy.ones((h, w, 3), dtype=numpy.uint8) * 255
                        trans_image = numpy.ones((h, w, 3), dtype=numpy.uint8) * 255
                        org_image[:h0, :w0, :] = self.__outImageRGB[:, :, :]
                        trans_image[:h1, :w1, :] = __rgbImg[:, :, :]
                        self.__outImageRGB = func(org_image, trans_image)
                        self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # 加
    def __addImage(self):
        self.__operation(cv2.add)

    # 减
    def __subtractImage(self):
        self.__operation(cv2.subtract)

    # 乘

    def __multiplyImage(self):
        self.__operation(cv2.multiply)

    # 自定义融合
    def __fusionImagezixuan(self):
        if self.__fileName:
            __fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image Files(*.png *.jpeg *.jpg *.bmp)')
            if __fileName and os.path.exists(__fileName):
                self.__bgrImg = cv2.imread(__fileName)
                self.__bgrImg = cv2.cvtColor(self.__bgrImg, cv2.COLOR_BGR2RGB)
                self.org_image = self.__outImageRGB.copy()
                self.trans_image = self.__bgrImg.copy()
                if self.__outImageRGB.shape != self.__bgrImg.shape:
                    # 图片尺寸相同才能进行运算
                    reply = QMessageBox.question(self, '确认完成操作吗', '提示：图片尺寸不一致，操作可能不好', QMessageBox.Yes,
                                                 QMessageBox.No)
                    if reply == QMessageBox.No:
                        return
                    else:
                        h0, w0 = self.__outImageRGB.shape[0], self.__outImageRGB.shape[1]  # cv2 读取出来的是h,w,c
                        h1, w1 = self.__bgrImg.shape[0], self.__bgrImg.shape[1]
                        h = max(h0, h1)
                        w = max(w0, w1)
                        self.org_image = numpy.ones((h, w, 3), dtype=numpy.uint8) * 255
                        self.trans_image = numpy.ones((h, w, 3), dtype=numpy.uint8) * 255
                        self.org_image[:h0, :w0, :] = self.__outImageRGB[:, :, :]
                        self.trans_image[:h1, :w1, :] = self.__bgrImg[:, :, :]
            else:
                return
            i = PySimpleGUI.popup_get_text('0-1之间', title='请输入后读入图所占的比重')
            if not i:
                return
            i=float(i)
            self.__outImageRGB = cv2.addWeighted(self.org_image, 1 - i, self.trans_image, i, 0)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # 1:1融合
    def __fusionImage(self):
        if self.__fileName:
            __fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '.',
                                                        'Image Files(*.png *.jpeg *.jpg *.bmp)')
            if __fileName and os.path.exists(__fileName):
                self.__bgrImg = cv2.imread(__fileName)
                # 图片尺寸相同才能进行运算
                self.__bgrImg = cv2.cvtColor(self.__bgrImg, cv2.COLOR_BGR2RGB)
                if self.__outImageRGB.shape != self.__bgrImg.shape:
                    # 图片尺寸相同才能进行运算
                    reply = QMessageBox.question(self, '确认完成操作吗', '提示：图片尺寸不一致，操作可能不好', QMessageBox.Yes,
                                                 QMessageBox.No)
                    if reply == QMessageBox.No:
                        return
                h0, w0 = self.__outImageRGB.shape[0], self.__outImageRGB.shape[1]  # cv2 读取出来的是h,w,c
                h1, w1 = self.__bgrImg.shape[0], self.__bgrImg.shape[1]
                h = max(h0, h1)
                w = max(w0, w1)
                org_image = numpy.ones((h, w, 3), dtype=numpy.uint8) * 255
                trans_image = numpy.ones((h, w, 3), dtype=numpy.uint8) * 255
                org_image[:h0, :w0, :] = self.__outImageRGB[:, :, :]
                trans_image[:h1, :w1, :] = self.__bgrImg[:, :, :]
                self.__outImageRGB = cv2.addWeighted(org_image, 0.5, trans_image, 0.5, 0)
                self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # 缩放调节子窗口
    def __openZoomWindow(self):
        self.__openPropertyWindow('缩放', self.__changeZoom)

    # 缩放
    def __changeZoom(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        # None的size是1 ！！！
        if numpy.size(__img) > 1:
            # 计算比例
            i = int(val)
            if i == -100:
                k = 0.01
            elif i >= 0:
                k = (i + 10) / 10
            else:
                k = (i + 100) / 100
            # 直接cv2.resize()缩放
            self.__tempImageRGB = cv2.resize(__img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # 旋转调节子窗口
    def __openRotateWindow(self):
        self.__openPropertyWindow('旋转', self.__changeRotate)
        if self.__fileName:
            # 重设属性值取值范围
            self.__propertyWindow.slider.setMaximum(360)
            self.__propertyWindow.slider.setMinimum(-360)
            self.__propertyWindow.spinBox.setMaximum(360)
            self.__propertyWindow.spinBox.setMinimum(-360)

    # 保存子函数
    def __baocuncaijian(self, x, y, w, h):
        img = self.__outImageRGB.copy()
        img = img[y:y + h, x:x + w]
        self.__outImageRGB = img.copy()
        self.__drawImage(self.outImageView, self.__outImageRGB)

    # 裁剪事件
    def __caijian(self):
        if self.__fileName:
            crop = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2BGR)
            roi = cv2.selectROI(windowName="original", img=crop, showCrosshair=True, fromCenter=False)
            x, y, w, h = roi
            # print(roi)
            cv2.destroyAllWindows()
            # 显示ROI并保存图片
            self.__baocuncaijian(x, y, w, h)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # 旋转
    def __changeRotate(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        # None的size是1 ！！！
        if numpy.size(__img) > 1:
            # 比例
            k = int(val)
            (h, w) = __img.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            # 绕图片中心旋转
            m = cv2.getRotationMatrix2D((cX, cY), k, 1.0)
            # 计算调整后的图片显示大小，使得图片不会被切掉边缘
            cos = numpy.abs(m[0, 0])
            sin = numpy.abs(m[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            m[0, 2] += (nW / 2) - cX
            m[1, 2] += (nH / 2) - cY
            # 变换，并设置旋转调整后产生的无效区域为白色
            self.__tempImageRGB = __img = cv2.warpAffine(__img, m, (nW, nH), borderValue=(255, 255, 255))
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # 垂直对称
    def __chuizhiduicheng(self):
        if self.__fileName:
            img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2BGR)
            v_flip = cv2.flip(img, 0)
            self.__outImageRGB = cv2.cvtColor(v_flip, cv2.COLOR_BGR2RGB)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # 水平对称
    def __shuipingduicheng(self):
        if self.__fileName:
            img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2BGR)
            v_flip = cv2.flip(img, 1)
            self.__outImageRGB = cv2.cvtColor(v_flip, cv2.COLOR_BGR2RGB)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
    #镜像对称
    def __jingxiangduicheng(self):
        if self.__fileName:
            img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2BGR)
            v_flip = cv2.flip(img, -1)
            self.__outImageRGB = cv2.cvtColor(v_flip, cv2.COLOR_BGR2RGB)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # -----------------------------------直方图均衡-----------------------------------
    # 绘制直方图
    def __histogram(self):
        if self.__fileName:
            # 如果是灰度图
            if len(self.__outImageRGB.shape) < 3:
                # __hist = cv2.calcHist([self.__outImageRGB], [0], None, [256], [0, 256])
                # __hist /= self.__outImageRGB.shape[0] * self.__outImageRGB.shape[1]
                # plt.plot(__hist)
                # 使用 matplotlib 的绘图功能同时绘制单通道的直方图
                # density的类型是 bool型，指定为True,则为频率直方图，反之为频数直方图
                plt.hist(self.__outImageRGB.ravel(), bins=255, rwidth=0.8, range=(0, 256), density=True)
            # 如果是RGB图
            else:
                color = {'r', 'g', 'b'}
                # 使用 matplotlib 的绘图功能同时绘制多通道 RGB 的直方图
                for i, col in enumerate(color):
                    __hist = cv2.calcHist([self.__outImageRGB], [i], None, [256], [0, 256])
                    __hist /= self.__outImageRGB.shape[0] * self.__outImageRGB.shape[1]
                    plt.plot(__hist, color=col)
            # x轴长度区间
            plt.xlim([0, 256])
            # 显示直方图
            plt.show()
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # 直方图均衡化
    def __histogramEqualization(self):
        if self.__fileName:
            # 如果是灰度图
            if len(self.__outImageRGB.shape) < 3:
                self.__outImageRGB = cv2.equalizeHist(self.__outImageRGB)
            # 如果是RGB图
            else:
                # 分解通道，各自均衡化，再合并通道
                (r, g, b) = cv2.split(self.__outImageRGB)
                rh = cv2.equalizeHist(r)
                gh = cv2.equalizeHist(g)
                bh = cv2.equalizeHist(b)
                self.__outImageRGB = cv2.merge((rh, gh, bh))
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # -----------------------------------特征描述-----------------------------------
    # 区域面积测量
    def __quyumianji(self):
        if self.__fileName:
            if self.tezhengflag == 0:
                reply = QMessageBox.question(self, '提示', '请确保图像中测量的目标仅有一个', QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.tezhengflag = 1
                    return
            self.tezhengflag = 1
            if self.__fileName and len(self.__outImageRGB.shape) > 2:
                # 灰度化使得三通道RGB图变成单通道灰度图
                img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            else:
                img = self.__outImageRGB.copy()
            try:
                ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
                contours, layer_num = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
                #只调用contours中的第一个元素，即一张画面只进行一个目标物体的检测
                area1 = cv2.contourArea(contours[0])
                # print(int(area1)) contourArea测量相对较不精准，
                area2 = stats[1][4]
                if area2 <= 10:
                    QMessageBox.information(self, '面积', '目标识别错误')
                    return
                QMessageBox.information(self, '面积', '目标的面积为%d(单位长度为一个像素点)' % area2)
            except:
                QMessageBox.information(self, '面积', '目标识别错误')
                return
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # 区域周长测量
    def __quyuzhouchang(self):
        if self.__fileName:
            if self.tezhengflag == 0:
                reply = QMessageBox.question(self, '提示', '请确保图像中测量的目标仅有一个', QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.tezhengflag = 1
                    return
            self.tezhengflag = 1
            if self.__fileName and len(self.__outImageRGB.shape) > 2:
                # 灰度化使得三通道RGB图变成单通道灰度图
                img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            else:
                img = self.__outImageRGB.copy()
            ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
            contours, layer_num = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            length = cv2.arcLength(contours[0], True)
            if length <= 10:
                QMessageBox.information(self, '周长', '目标识别错误')
                return
            QMessageBox.information(self, '周长', '目标的周长为%d(单位长度为一个像素点)' % length)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')

    # 区域圆形度测量
    def __quyuyuanxingdu(self):
        if self.__fileName:
            if self.tezhengflag == 0:
                reply = QMessageBox.question(self, '提示', '请确保图像中测量的目标仅有一个', QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.tezhengflag = 1
                    return
            self.tezhengflag = 1
            if self.__fileName and len(self.__outImageRGB.shape) > 2:
                # 灰度化使得三通道RGB图变成单通道灰度图
                img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            else:
                img = self.__outImageRGB.copy()
            ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
            contours, layer_num = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            length = cv2.arcLength(contours[0], True)
            area = stats[1][4]
            yuanxingdu = (4 * math.pi * area) / (length * length)
            if length <= 10 or area <= 10 or yuanxingdu > 2:
                QMessageBox.information(self, '圆形度', '目标识别错误')
                return
            if yuanxingdu >= 0.99:
                QMessageBox.information(self, '圆形度', '目标是圆形，圆形度为1')
                return
            QMessageBox.information(self, '圆形度', '目标的圆形度为%.2f' % yuanxingdu)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return
    def __quyuqiuzhuangdu(self):
        if self.__fileName:
            try:
                if self.tezhengflag == 0:
                    reply = QMessageBox.question(self, '提示', '请确保图像中测量的目标仅有一个', QMessageBox.Yes, QMessageBox.No)
                    if reply == QMessageBox.No:
                        self.tezhengflag = 1
                        return
                self.tezhengflag = 1
                if self.__fileName and len(self.__outImageRGB.shape) > 2:
                    # 灰度化使得三通道RGB图变成单通道灰度图
                    img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
                else:
                    img = self.__outImageRGB.copy()
                ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
                contours, layer_num = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
                length = cv2.arcLength(contours[0], True)
                area = stats[1][4]
                raw_dist = numpy.empty(img.shape, dtype=numpy.float32)
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        raw_dist[i, j] = cv2.pointPolygonTest(contours[0], (j, i), True)
                # 获取最大值即内接圆半径
                minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
                maxVal=abs(maxVal)
                # print(maxVal)
                Rmin=length/(2 * math.pi)
                # print(Rmin)
                qiuzhuangdu=maxVal/Rmin
                if length <= 10 or area <= 10 or qiuzhuangdu > 2:
                    QMessageBox.information(self, '球状度', '目标识别错误')
                    return
                if qiuzhuangdu >= 0.99:
                    QMessageBox.information(self, '球状度', '目标是圆形，球状度为1')
                    return
                QMessageBox.information(self, '球状度', '目标的球状度为%.2f' % qiuzhuangdu)
            except:
                QMessageBox.information(self, '球状度', '目标识别错误')
                return
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return
    # 区域矩形面积测量
    def __quyujuxing(self):
        if self.__fileName:
            if self.tezhengflag == 0:
                reply = QMessageBox.question(self, '提示', '请确保图像中测量的目标仅有一个', QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.tezhengflag = 1
                    return
            self.tezhengflag = 1
            if self.__fileName and len(self.__outImageRGB.shape) > 2:
                # 灰度化使得三通道RGB图变成单通道灰度图
                img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            else:
                img = self.__outImageRGB.copy()
            ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
            contours, layer_num = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            # 最小外接矩形
            min_rect = cv2.minAreaRect(contours[0])  # 返回的是一个元组，第一个元素是左上角点坐标组成的元组，第二个元素是矩形宽高组成的元组，第三个是旋转的角度
            box = cv2.boxPoints(min_rect)  # 返回的是一个numpy矩阵
            min_rect_area = cv2.contourArea(box)
            area2 = stats[1][4]
            if min_rect_area <= 10:
                QMessageBox.information(self, '最小外接矩形面积', '目标识别错误')
                return
            elif min_rect_area < area2:
                min_rect_area = area2
            QMessageBox.information(self, '最小外接矩形面积', '面积为%d(单位长度为一个像素点)' % min_rect_area)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 区域细长度测量
    def __quyujuxingdu(self):
        if self.__fileName:
            if self.tezhengflag == 0:
                reply = QMessageBox.question(self, '提示', '请确保图像中测量的目标仅有一个', QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.tezhengflag = 1
                    return
            self.tezhengflag = 1
            if self.__fileName and len(self.__outImageRGB.shape) > 2:
                # 灰度化使得三通道RGB图变成单通道灰度图
                img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            else:
                img = self.__outImageRGB.copy()
            ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
            contours, layer_num = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            # 最小外接矩形
            min_rect = cv2.minAreaRect(contours[0])  # 返回的是一个元组，第一个元素是左上角点坐标组成的元组，第二个元素是矩形宽高组成的元组，第三个是旋转的角度
            box = cv2.boxPoints(min_rect)  # 返回的是一个numpy矩阵
            min_rect_area = cv2.contourArea(box)
            area2 = stats[1][4]
            ee = area2 / (min_rect_area + 0.01)
            if ee >= 1 and ee < 2:
                QMessageBox.information(self, '矩形度', '该图形是矩形,矩形度为1')
                return
            elif ee >= 2:
                QMessageBox.information(self, '矩形度', '目标识别错误')
                return
            QMessageBox.information(self, '矩形度', '矩形度为%.2f' % ee)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    def __quyuxichangdu(self):
        if self.__fileName:
            if self.tezhengflag == 0:
                reply = QMessageBox.question(self, '提示', '请确保图像中测量的目标仅有一个', QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.tezhengflag = 1
                    return
            self.tezhengflag = 1
            if self.__fileName and len(self.__outImageRGB.shape) > 2:
                # 灰度化使得三通道RGB图变成单通道灰度图
                img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            else:
                img = self.__outImageRGB.copy()
            ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
            contours, layer_num = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            # 最小外接矩形
            min_rect = cv2.minAreaRect(contours[0])  # 返回的是一个元组，第一个元素是左上角点坐标组成的元组，第二个元素是矩形宽高组成的元组，第三个是旋转的角度
            box = cv2.boxPoints(min_rect)  # 返回的是一个numpy矩阵
            min_rect_h = min_rect[1][0]
            min_rect_w = min_rect[1][1]
            if min_rect_w <= 10:
                QMessageBox.information(self, '区域细长度', '目标识别错误')
                return
            e = min_rect_h / min_rect_w
            if e <= 0 or e > 1e8:
                QMessageBox.information(self, '区域细长度', '目标识别错误')
                return
            QMessageBox.information(self, '区域细长度', '细长度为%.2f' % e)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    def __quyuzhongxin(self):
        if self.__fileName:
            if self.tezhengflag == 0:
                reply = QMessageBox.question(self, '提示', '请确保图像中测量的目标仅有一个', QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.tezhengflag = 1
                    return
            self.tezhengflag = 1
            if self.__fileName and len(self.__outImageRGB.shape) > 2:
                # 灰度化使得三通道RGB图变成单通道灰度图
                img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            else:
                img = self.__outImageRGB.copy()
            try:
                ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
                contours, layer_num = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
                area2 = stats[1][4]
                length = cv2.arcLength(contours[0], True)
                if area2 <= 10 or length <= 10:
                    QMessageBox.information(self, '重心', '目标识别错误')
                    return
                centroid = centroids[0]
                a = int(centroid[0])
                b = int(centroid[1])
                QMessageBox.information(self, '重心', '重心坐标为(%d,%d)' % (a, b))
            except:
                QMessageBox.information(self, '重心', '区域识别错误')
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # Harris角(角点检测)
    def __quyuHarris(self):
        if self.__fileName:
            gray = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            gray = numpy.float32(gray)
            # 输入图像必须是float32，最后一个参数在0.04到0.05
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            # Threshold for an optimal value, it may vary depending on the image.
            img = self.__outImageRGB.copy()
            img[dst > 0.01 * dst.max()] = [255, 0, 0]
            self.__outImageRGB = img.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    def __quyusubHarris(self):
        if self.__fileName:
            img = self.__outImageRGB.copy()
            gray = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            gray = numpy.float32(gray)
            # 输入图像必须是float32，最后一个参数在0.04到0.05
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
            dst = numpy.uint8(dst)
            # find centroids
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            # define the criteria to stop and refine the corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(gray, numpy.float32(centroids), (5, 5), (-1, -1), criteria)
            # Now draw them
            res = numpy.hstack((centroids, corners))
            res = numpy.int0(res)
            img[res[:, 1], res[:, 0]] = [0, 0, 255]
            img[res[:, 3], res[:, 2]] = [0, 255, 0]
            self.__outImageRGB = img.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    def __quyujianceyuan(self):
        if self.__fileName:
            try:
                image = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_BGR2RGB)
                image_copy = image.copy()
                # 灰度处理
                # cv2.imshow('binary', th2)

                filter = cv2.pyrMeanShiftFiltering(image, 10, 40)

                # 转换成灰度图
                filter_gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)

                # 霍夫曼圆圈检测
                circles = cv2.HoughCircles(filter_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=100, minRadius=0,
                                           maxRadius=0)
                circles = numpy.uint16(numpy.around(circles))
                # 遍历
                for circle in circles[0, :]:
                    cv2.circle(image_copy, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
                    cv2.circle(image_copy, (circle[0], circle[1]), 2, (255, 0, 0), 2)
                self.__outImageRGB = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
                self.__drawImage(self.outImageView, self.__outImageRGB)
            except:
                QMessageBox.information(self, '检测圆', '识别圆失败！')
                return
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 轮廓识别
    def __quyulunkuoshibie(self):
        self.shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'multiples': 0, 'circles': 0}
        if self.__fileName:
            try:
                QMessageBox.information(self, '提示（此模块还不完善）', '请提供轮廓清晰的图片')
                if len(self.__outImageRGB.shape) > 2:
                    # 灰度化使得三通道RGB图变成单通道灰度图
                    gray = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
                else:
                    gray = self.__outImageRGB.copy()
                gray=cv2.medianBlur(gray, 5)
                ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                contours, layer_num = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for i in range(len(contours) - 1):
                    ep = 0.01 * cv2.arcLength(contours[i], True)
                    ap = cv2.approxPolyDP(contours[i], ep, True)
                    co = len(ap)
                    if co == 3:
                        st = '三⾓形'
                        self.shapes['triangle'] += 1
                    elif co == 4:
                        st = '矩形'
                        self.shapes['rectangle'] += 1
                    elif co == 10:
                        st = '五⾓星'
                        self.shapes['polygons'] += 1
                    else:
                        st = '其他'
                        self.shapes['circles'] += 1
                QMessageBox.information(self, '轮廓识别', '识别结果为:三角形有%d个，\
                \n矩形有%d个，\n五角星有%d个，\n圆有%d个' % (
                    self.shapes['triangle'], self.shapes['rectangle'], self.shapes['polygons'], self.shapes['circles']))
            except:
                QMessageBox.information(self, '轮廓识别', '目标识别错误')
                return
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # -----------------------------------图像截取-----------------------------------
    # 迭代算法的实现函数
    def Iterate_Thresh(self, img, initval, MaxIterTimes=20, thre=1):
        """ 阈值迭代算法
         Args:
          img: 灰度图像
          initval: 初始阈值
          MaxIterTimes: 最⼤迭代次数，默认20
          thre：临界差值，默认为1
         Return:
        计算出的阈值
         """
        mask1, mask2 = (img > initval), (img <= initval)
        T1 = numpy.sum(mask1 * img) / numpy.sum(mask1)
        T2 = numpy.sum(mask2 * img) / numpy.sum(mask2)
        T = (T1 + T2) / 2
        # 终⽌条件

        if abs(T - initval) < thre or MaxIterTimes == 0:
            return T
        return self.Iterate_Thresh(img, T, MaxIterTimes - 1)

    def __diedai(self):
        if self.__fileName:
            if len(self.__outImageRGB.shape) == 3:
                reply = QMessageBox.question(self, '确认完成迭代操作吗', '提示：当前图片不是灰度图哦，迭代效果可能不好', QMessageBox.Yes,
                                             QMessageBox.No)
                if reply == QMessageBox.No:
                    return
                else:
                    initthre = numpy.mean(self.__outImageRGB)
                    # 阈值迭代
                    thresh = self.Iterate_Thresh(self.__outImageRGB, initthre, 50)
                    dst = cv2.threshold(self.__outImageRGB, thresh, 255, cv2.THRESH_BINARY)[1]
                    self.__outImageRGB = dst.copy()
                    self.__drawImage(self.outImageView, self.__outImageRGB)
            else:
                initthre = numpy.mean(self.__outImageRGB)
                # 阈值迭代
                thresh = self.Iterate_Thresh(self.__outImageRGB, initthre, 50)
                dst = cv2.threshold(self.__outImageRGB, thresh, 255, cv2.THRESH_BINARY)[1]
                self.__outImageRGB = dst.copy()
                self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 区域增长算法的实现函数
    def regionGrow(self, gray, seeds, thresh, p):  # thresh表示与领域的相似距离，小于该距离就合并
        seedMark = numpy.zeros(gray.shape)
        # 八邻域
        if p == 8:
            connection = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        # 四邻域
        elif p == 4:
            connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        else:
            connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        # seeds内无元素时候生长停止
        while len(seeds) != 0:
            # 栈顶元素出栈
            pt = seeds.pop(0)
            for i in range(p):
                tmpX = int(pt[0] + connection[i][0])
                tmpY = int(pt[1] + connection[i][1])

                # 检测边界点
                if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
                    continue

                if abs(int(gray[tmpX, tmpY]) - int(gray[pt])) < thresh and seedMark[tmpX, tmpY] == 0:
                    seedMark[tmpX, tmpY] = 255
                    seeds.append((tmpX, tmpY))
        return seedMark

    def get_x_y(self, n):  # path表示图片路径，n表示要获取的坐标个数
        im = self.__outImageRGB
        plt.imshow(im, cmap=plt.get_cmap("gray"))
        pos = plt.ginput(n)
        return pos  # 得到的pos是列表中包含多个坐标元组

    def get_color(self, n):  # path表示图片路径，n表示要获取的坐标个数
        self.im = cv2.imread('11.jpg')
        plt.imshow(self.im)
        pos = plt.ginput(n)
        return pos  # 得到的pos是列表中包含多个坐标元组

    def __zhongzi(self):
        if self.__fileName:
            w = PySimpleGUI.popup_get_text('大于0的整数', title='选取种子的个数')
            if not w:
                return
            i = int(w)
            if i <= 0:
                return
            gray = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_BGR2GRAY)
            seeds = self.get_x_y(n=i)  # 获取初始种子
            # print("选取的初始点为：")
            new_seeds = []
            for seed in seeds:
                # print(seed)
                # 下面是需要注意的一点
                # 第一： 用鼠标选取的坐标为float类型，需要转为int型
                # 第二：用鼠标选取的坐标为（W,H），而我们使用函数读取到的图片是（行，列），而这对应到原图是（H,W），所以这里需要调换一下坐标位置，这是很多人容易忽略的一点
                new_seeds.append((int(seed[1]), int(seed[0])))  #
            result = self.regionGrow(gray, new_seeds, 3, 8)
            result2 = Image.fromarray(numpy.uint8(result))
            # result2.show()
            img2 = cv2.cvtColor(numpy.asarray(result2), cv2.COLOR_RGB2BGR)
            # cv2.imshow("gray mask", img2)
            # cv2.waitKey(0)
            h = self.__outImageRGB.shape[0]
            w = self.__outImageRGB.shape[1]
            # 把灰白图中获取到的掩模信息保存到彩色图中，实现彩色图的掩模
            for i in range(h):
                for j in range(w):
                    if result[i][j] != 255:
                        self.__outImageRGB[i][j][0] = 0
                        self.__outImageRGB[i][j][1] = 0
                        self.__outImageRGB[i][j][2] = 0
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    def __renlianjiequ(self):
        if self.__fileName:
            try:
                img = self.__outImageRGB.copy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # OpenCV人脸识别分类器
                classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                if not classifier:
                    QMessageBox.information(self, '提示', '依赖未配置，本功能暂时不可用')
                    return
                color = (0, 255, 0)  # 定义绘制颜色
                # 调用识别人脸
                faceRects = classifier.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
                if len(faceRects):  # 大于0则检测到人脸
                    for faceRect in faceRects:  # 单独框出每一张人脸
                        x, y, w, h = faceRect
                        # 框出人脸
                        cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
                        # # 左眼
                        # cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
                        # # 右眼
                        # cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
                        # # 嘴巴
                        # cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), color)
                self.__outImageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.__drawImage(self.outImageView, self.__outImageRGB)
            except:
                QMessageBox.information(self, '提示', '画面中人脸特征不够明显！(或依赖未配置)')
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    def __renlianjiequ2(self):
        if self.__fileName:
            try:
                reply = QMessageBox.question(self, '确认完成勾勒操作吗', '提示：请输入单张清晰的人脸素材图', QMessageBox.Yes,
                                             QMessageBox.No)
                if reply == QMessageBox.No:
                    return
                face_image = self.__outImageRGB.copy()
                #Get the face landmarks list 获取特征列表
                face_landmarks_list = face_recognition.face_landmarks(face_image)
                # print the face landmarks list
                print(face_landmarks_list)
                # STEP4: 循环遍历，分离列表中各个通道的值。Loop around to convert to draw objects
                for face_landmarks in face_landmarks_list:
                    # convert the numpy array image into pil image object
                    pil_image = Image.fromarray(face_image)
                    # convert the pil image to draw object
                    draw_face_landmark = ImageDraw.Draw(pil_image)
                    # join each face landmark points
                    draw_face_landmark.line(face_landmarks['chin'], fill=(255, 255, 255), width=2)
                    draw_face_landmark.line(face_landmarks['left_eyebrow'], fill=(255, 255, 255), width=2)
                    draw_face_landmark.line(face_landmarks['right_eyebrow'], fill=(255, 255, 255), width=2)
                    draw_face_landmark.line(face_landmarks['nose_bridge'], fill=(255, 255, 255), width=2)
                    draw_face_landmark.line(face_landmarks['nose_tip'], fill=(255, 255, 255), width=2)
                    draw_face_landmark.line(face_landmarks['left_eye'], fill=(255, 255, 255), width=2)
                    draw_face_landmark.line(face_landmarks['right_eye'], fill=(255, 255, 255), width=2)
                    draw_face_landmark.line(face_landmarks['top_lip'], fill=(255, 255, 255), width=2)
                    draw_face_landmark.line(face_landmarks['bottom_lip'], fill=(255, 255, 255), width=2)
                img = cv2.cvtColor(numpy.asarray(pil_image), cv2.COLOR_RGB2BGR)
                self.__outImageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.__drawImage(self.outImageView, self.__outImageRGB)
            except:
                QMessageBox.information(self, '提示', '画面中人脸特征不够明显！')
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # -----------------------------------噪声-----------------------------------
    # 加高斯噪声
    def __addGasussNoise(self):
        if self.__fileName:
            # 图片灰度标准化
            self.__outImageRGB = numpy.array(self.__outImageRGB / 255, dtype=float)
            # 产生高斯噪声
            noise = numpy.random.normal(0, 0.001 ** 0.5, self.__outImageRGB.shape)
            # 叠加图片和噪声
            out = cv2.add(self.__outImageRGB, noise)
            # 还原灰度并截取灰度区间
            self.__outImageRGB = numpy.clip(numpy.uint8(out * 255), 0, 255)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 加椒盐噪声
    def __addUniformNoise(self):
        if self.__fileName:
            # 起始范围
            low = 100
            # 终止范围
            height = 150
            # 搞一个与图片同规模数组
            out = numpy.zeros(self.__outImageRGB.shape, numpy.uint8)
            # 噪声生成比率
            ratio = 0.05
            # 遍历图片
            for i in range(self.__outImageRGB.shape[0]):
                for j in range(self.__outImageRGB.shape[1]):
                    # 随机数[0.0,1.0)
                    r = random.random()
                    # 填充黑点
                    if r < ratio:
                        # 生成[low，height]的随机值
                        out[i][j] = random.randint(low, height)
                    # 填充白点
                    elif r > 1 - ratio:
                        out[i][j] = random.randint(low, height)
                    # 填充原图
                    else:
                        out[i][j] = self.__outImageRGB[i][j]
            self.__outImageRGB = out.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 加随机噪声
    def __addImpulseNoise(self):
        if self.__fileName:
            # 搞一个与图片同规模的数组
            out = numpy.zeros(self.__outImageRGB.shape, numpy.uint8)
            # 椒盐噪声生成比率
            ratio = 0.05
            # 遍历图片
            for i in range(self.__outImageRGB.shape[0]):
                for j in range(self.__outImageRGB.shape[1]):
                    # 随机数[0.0,1.0)
                    r = random.random()
                    # 填充黑点
                    if r < ratio:
                        out[i][j] = 0
                    # 填充白点
                    elif r > 1 - ratio:
                        out[i][j] = 255
                    # 填充原图
                    else:
                        out[i][j] = self.__outImageRGB[i][j]
            self.__outImageRGB = out.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 加泊松噪声
    def __addbosongNoise(self):
        if self.__fileName:
            # 搞一个与图片同规模的噪声数组
            noise_type = numpy.random.poisson(lam=0.55, size=self.__outImageRGB.shape).astype(
                dtype='uint8')  # lam>=0 值越小，噪声频率就越少，size为图像尺寸
            out = self.__outImageRGB + noise_type
            self.__outImageRGB = out.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # -----------------------------------空域滤波-----------------------------------
    # 均值滤波
    def __meanValueFilter(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.blur(self.__outImageRGB, (5, 5))
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 中值滤波
    def __medianValueFilter(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.medianBlur(self.__outImageRGB, 5)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 自实现中值滤波
    def __zishixianmedianValueFilter(self):
        if self.__fileName:
            filiter_size = 3
            img = self.__outImageRGB.copy()
            image_copy = numpy.array(img, copy=True).astype(numpy.float32)
            processed = numpy.zeros_like(image_copy)
            middle = int(filiter_size / 2)
            r = numpy.zeros(filiter_size * filiter_size)
            g = numpy.zeros(filiter_size * filiter_size)
            b = numpy.zeros(filiter_size * filiter_size)

            for i in range(middle, image_copy.shape[0] - middle):
                for j in range(middle, image_copy.shape[1] - middle):
                    count = 0
                    # 依次取出模板中对应的像素值
                    for m in range(i - middle, i + middle + 1):
                        for n in range(j - middle, j + middle + 1):
                            r[count] = image_copy[m][n][0]
                            g[count] = image_copy[m][n][1]
                            b[count] = image_copy[m][n][2]
                            count += 1
                    r.sort()
                    g.sort()
                    b.sort()
                    processed[i][j][0] = r[int(filiter_size * filiter_size / 2)]
                    processed[i][j][1] = g[int(filiter_size * filiter_size / 2)]
                    processed[i][j][2] = b[int(filiter_size * filiter_size / 2)]
            processed = numpy.clip(processed, 0, 255).astype(numpy.uint8)
            self.__outImageRGB = processed.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 高斯滤波
    def __guasslvbo(self):
        if self.__fileName:
            self.__outImageRGB = cv2.GaussianBlur(self.__outImageRGB, (5, 5), 0)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # Sobel算子锐化
    def __sobel(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.Sobel(self.__outImageRGB, -1, 1, 1, 3)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # Prewitt算子锐化
    def __prewitt(self):
        if self.__fileName:
            # Prewitt 算子
            kernelx = numpy.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
            kernely = numpy.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
            # 通过自定义卷积核实现卷积
            imgx = cv2.filter2D(self.__outImageRGB, -1, kernelx)
            imgy = cv2.filter2D(self.__outImageRGB, -1, kernely)
            # 合并
            self.__outImageRGB = cv2.add(imgx, imgy)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 拉普拉斯算子锐化
    def __laplacian(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.Laplacian(self.__outImageRGB, -1, ksize=3)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    def __CannyAction(self):
        if self.__fileName:
            # 直接调库
            w = PySimpleGUI.popup_get_text('大于0，小于255的整数', title='请输入Canny算子的阈值下限')
            if not w:
                return
            # 最小阈值越大，介于两阈值之间但靠近边界的许多点被舍弃，会造成边缘的破损，细节相对减少
            wmin = int(w)
            # 最大阈值越大，直接舍弃掉的点越多，这些舍弃是大面积的，同样使细节减少，突出更明显的边缘
            w = PySimpleGUI.popup_get_text('大于最小阈值，小于255的整数', title='请输入Canny算子的阈值上限')
            if not w:
                return
            wmax = int(w)
            self.__outImageRGB = cv2.Canny(self.__outImageRGB, wmin, wmax)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # -----------------------------------美化效果-----------------------------------
    # 浮雕
    def __fudiao(self):
        if self.__fileName:
            def Filter_Fudiao(src_img):
                # filter=np.array([[-1,0,0],[0,0,0],[0,0,1]])
                filter = numpy.array([[-1, 0], [0, 1]])
                row = src_img.shape[0]
                col = src_img.shape[1]
                new_img = numpy.zeros([row, col], dtype=numpy.uint8)
                for i in range(row - 1):
                    for j in range(col - 1):
                        new_value = numpy.sum(src_img[i:i + 2, j:j + 2] * filter) + 128  # point multiply
                        if new_value > 255:
                            new_value = 255
                        elif new_value < 0:
                            new_value = 0
                        else:
                            pass
                        new_img[i, j] = new_value
                return new_img

            gray_img = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_BGR2GRAY)
            new_img = Filter_Fudiao(gray_img)
            self.__outImageRGB = new_img.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 毛玻璃
    def __maoboli(self):
        if self.__fileName:
            dst = numpy.zeros_like(self.__outImageRGB)
            # 获取图像行和列
            rows, cols = self.__outImageRGB.shape[:2]
            # 定义偏移量和随机数
            offsets = 5
            random_num = 0
            # 毛玻璃效果: 像素点邻域内随机像素点的颜色替代当前像素点的颜色
            for y in range(rows - offsets):
                for x in range(cols - offsets):
                    random_num = numpy.random.randint(0, offsets)
                    dst[y, x] = self.__outImageRGB[y + random_num, x + random_num]
            self.__outImageRGB = dst.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 卡通化
    def __katonghua(self):
        if self.__fileName:
            def edge_mask(img, line_size, blur_value):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_blur = cv2.medianBlur(gray, blur_value)
                edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size,
                                              blur_value)
                return edges

            def color_quantization(img, k):
                # Transform the image
                data = numpy.float32(img).reshape((-1, 3))

                # Determine criteria
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

                # Implementing K-Means
                ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                center = numpy.uint8(center)
                result = center[label.flatten()]
                result = result.reshape(img.shape)
                return result

            line_size = 7
            blur_value = 7
            edges = edge_mask(self.__outImageRGB, line_size, blur_value)
            total_color = 9
            img = color_quantization(self.__outImageRGB, total_color)
            blurred = cv2.bilateralFilter(img, d=7,
                                          sigmaColor=200, sigmaSpace=200)
            cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
            self.__outImageRGB = cartoon.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    def __masaike(self):
        if self.__fileName:
            crop = self.__outImageRGB.copy()
            if True :
                def do_mosaic(img, x, y, w, h, neighbor=9):
                    """
                    马赛克单独实现
                    :param img
                    :param int x :  马赛克左顶点
                    :param int y:  马赛克左顶点
                    :param int w:  马赛克宽
                    :param int h:  马赛克高
                    :param int neighbor:  马赛克每一块的宽
                    """
                    for i in range(0, h, neighbor):
                        for j in range(0, w, neighbor):
                            rect = [j + x, i + y]
                            color = img[i + y][j + x].tolist()  # 关键点1 tolist
                            left_up = (rect[0], rect[1])
                            x2 = rect[0] + neighbor - 1  # 关键点2 减去一个像素
                            y2 = rect[1] + neighbor - 1
                            if x2 > x + w:
                                x2 = x + w
                            if y2 > y + h:
                                y2 = y + h
                            right_down = (x2, y2)
                            cv2.rectangle(img, left_up, right_down, color, -1)  # 替换为为一个颜值值
                    return img
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                roi = cv2.selectROI(windowName="original", img=crop, showCrosshair=True, fromCenter=False)
                x, y, w, h = roi
                cv2.destroyAllWindows()
                img_mosaic = do_mosaic(crop, x, y, w, h)
                img_mosaic = cv2.cvtColor(img_mosaic, cv2.COLOR_BGR2RGB)
                self.__outImageRGB = img_mosaic.copy()
                self.__drawImage(self.outImageView, self.__outImageRGB)
                return
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 加框
    def __jiakuang(self):
        color0 = [255, 0, 0]
        if self.__fileName:
            def color(value):
                digit = list(map(str, range(10))) + list("abcdef")
                if isinstance(value, tuple):
                    string = '#'
                    for i in value:
                        a1 = i // 16
                        a2 = i % 16
                        string += digit[a1] + digit[a2]
                        return string
                elif isinstance(value, str):
                    a1 = digit.index(value[1]) * 16 + digit.index(value[2])
                    a2 = digit.index(value[3]) * 16 + digit.index(value[4])
                    a3 = digit.index(value[5]) * 16 + digit.index(value[6])
                    return (a1, a2, a3)
            try:
                # pos = self.get_color(1)
                # i = int(pos[0][0])
                # j = int(pos[0][1])
                # color0 = self.im[j][i]
                col = QColorDialog.getColor()
                self.frame = QFrame(self)
                # 检测用的选择是否合法(点击cancel就是非法,否则就是合法)
                if col.isValid():
                    color0=color(col.name())
                else:
                    return
                print(col)
                img = cv2.copyMakeBorder(self.__outImageRGB, 20, 20, 20, 20, cv2.BORDER_CONSTANT,
                                         value=[int(str(color0[0]).strip()), int(str(color0[1]).strip()),
                                                int(str(color0[2]).strip())])
                self.__outImageRGB = img.copy()
                self.__drawImage(self.outImageView, self.__outImageRGB)
            except:
                QMessageBox.information(self, '提示', '边框设置失败')
                return
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    def __ronghekuang(self):
        if self.__fileName:
            ak1 = self.__outImageRGB.copy()
            ak2 = self.__outImageRGB.copy()
            h, w, s = ak1.shape
            __fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image Files(*.png *.jpeg *.jpg *.bmp)')
            # 文件存在
            if __fileName and os.path.exists(__fileName):
                # 设置打开的文件名属性
                self.__fileName = __fileName
                # 读入，记得转换颜色空间！！！
                __bgrImg = cv2.imread(self.__fileName)
                ak2 = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
            else:
                return
            ak2 = cv2.resize(ak2, (w, h), interpolation=cv2.INTER_LINEAR)
            b, g, r = cv2.split(ak2)
            b_a = numpy.asarray(b)
            g_a = numpy.asarray(g)
            r_a = numpy.asarray(r)
            mc = numpy.zeros((h, w))
            for i in range(h):
                for j in range(w):
                    mc[i][j] = int(b_a[i][j]) + int(g_a[i][j]) + int(r_a[i][j])

            h, w = mc.shape
            avarege = int((numpy.sum(mc) / (w * h)) * 0.75)
            gatesize = avarege
            for i in range(h):
                for j in range(w):
                    if mc[i][j] >= gatesize:
                        ak2[i][j][0] = ak1[i][j][0]
                        ak2[i][j][1] = ak1[i][j][1]
                        ak2[i][j][2] = ak1[i][j][2]
            self.__outImageRGB = ak2.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    def __pingtu(self):
        if self.__fileName:
            # 打开文件选择窗口
            __fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image Files(*.png *.jpeg *.jpg *.bmp)')
            # 文件存在
            if __fileName and os.path.exists(__fileName):
                # 设置打开的文件名属性
                self.__fileName = __fileName
                # 读入，记得转换颜色空间！！！
                __bgrImg = cv2.imread(self.__fileName)
                __bgrImg = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
                h0, w0 = self.__outImageRGB.shape[0], self.__outImageRGB.shape[1]  # cv2 读取出来的是h,w,c
                h1, w1 = __bgrImg.shape[0], __bgrImg.shape[1]
                h = max(h0, h1)
                w = max(w0, w1)
                org_image = numpy.ones((h, w, 3), dtype=numpy.uint8) * 255
                trans_image = numpy.ones((h, w, 3), dtype=numpy.uint8) * 255

                org_image[:h0, :w0, :] = self.__outImageRGB[:, :, :]
                trans_image[:h1, :w1, :] = __bgrImg[:, :, :]
                w = PySimpleGUI.popup_get_text('垂直或水平拼接', title='输入拼接方式')
                if not w:
                    return
                if w == '垂直':
                    img = numpy.concatenate((org_image, trans_image), 0)
                elif w == '水平':
                    img = numpy.concatenate((org_image, trans_image), 1)
                else:
                    return
                self.__outImageRGB = img.copy()
                self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # -----------------------------------相片滤镜-----------------------------------
    # 怀旧
    def __huaijiu(self):
        if self.__fileName:
            img = self.__outImageRGB.copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            height, width, n = img.shape
            img2 = img.copy()
            for i in range(height):
                for j in range(width):
                    b = img[i, j][0]
                    g = img[i, j][1]
                    r = img[i, j][2]
                    # 计算新的图像中的RGB值
                    B = int(0.273 * r + 0.535 * g + 0.131 * b)
                    G = int(0.347 * r + 0.683 * g + 0.167 * b)
                    R = int(0.395 * r + 0.763 * g + 0.188 * b)  # 约束图像像素值，防止溢出
                    img2[i, j][0] = max(0, min(B, 255))
                    img2[i, j][1] = max(0, min(G, 255))
                    img2[i, j][2] = max(0, min(R, 255))
                # 显示图像
                self.__outImageRGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 光晕
    def __guangyun(self):
        if self.__fileName:
            img = self.__outImageRGB.copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            rows, cols = img.shape[:2]
            centerX = rows / 2 - 20
            centerY = cols / 2 + 20
            radius = min(centerX, centerY)
            strength = 100
            dst = numpy.zeros((rows, cols, 3), dtype="uint8")
            for i in range(rows):
                for j in range(cols):
                    # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
                    distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
                    # 获取原始图像
                    B = img[i, j][0]
                    G = img[i, j][1]
                    R = img[i, j][2]
                    if distance < radius * radius:
                        # 按照距离大小计算增强的光照值
                        result = (int)(strength * (1.0 - math.sqrt(distance) / radius))
                        B = img[i, j][0] + result
                        G = img[i, j][1] + result
                        R = img[i, j][2] + result
                        # 判断边界 防止越界
                        B = min(255, max(0, B))
                        G = min(255, max(0, G))
                        R = min(255, max(0, R))
                        dst[i, j] = numpy.uint8((B, G, R))
                    else:
                        dst[i, j] = numpy.uint8((B, G, R))
            # 显示图像
            self.__outImageRGB = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # 流年
    def __liunian(self):
        if self.__fileName:
            img = self.__outImageRGB.copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            rows, cols = img.shape[:2]
            dst = numpy.zeros((rows, cols, 3), dtype="uint8")
            for i in range(rows):
                for j in range(cols):
                    B = math.sqrt(img[i, j][0]) * 12
                    G = img[i, j][1]
                    R = img[i, j][2]
                    if B > 255:
                        B = 255
                    dst[i, j] = numpy.uint8((B, G, R))
            # 显示图像
            self.__outImageRGB = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return
    #一键美化
    def __renlianmeihua(self):
        if self.__fileName:
            img=self.__outImageRGB.copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            # 调大亮度
            brightness = 1.1
            enh_bri = ImageEnhance.Brightness(image)
            image1 = enh_bri.enhance(brightness)
            #调大对比度
            contrast =1.1
            enh_con = ImageEnhance.Contrast(image1)
            image2 = enh_con.enhance(contrast)
            #调大饱和度
            color=1.1
            enh_col = ImageEnhance.Color(image2)
            image3 = enh_col.enhance(color)
            #调大清晰度
            sharpness =1.1
            enh_sha = ImageEnhance.Sharpness(image3)
            image4 = enh_sha.enhance(sharpness)
            #磨皮
            image5= cv2.cvtColor(numpy.asarray(image4), cv2.COLOR_RGB2BGR)
            image6 = cv2.bilateralFilter(image5, 0, 0, 10)
            self.__outImageRGB=cv2.cvtColor(image6, cv2.COLOR_BGR2RGB)
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return
    def __renlianmakeup(self):
        if self.__fileName:
            try:
                face_image = self.__outImageRGB.copy()
                face_landmarks_list = face_recognition.face_landmarks(face_image)
                # STEP4: Loop around to convert to draw objects
                for face_landmarks in face_landmarks_list:
                    # convert the numpy array image into pil image object
                    pil_image = Image.fromarray(face_image)
                    # convert the pil image to draw object
                    draw_landmark_for_makeup = ImageDraw.Draw(pil_image, "RGBA")
                    # draw the shapes and fill with color
                    # Make left, right eyebrows darker
                    # Polygon on top and line on bottom with dark color
                    draw_landmark_for_makeup.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
                    draw_landmark_for_makeup.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
                    draw_landmark_for_makeup.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
                    draw_landmark_for_makeup.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
                    # Add lipstick to top and bottom lips
                    # using red polygons and lines filled with red
                    draw_landmark_for_makeup.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
                    draw_landmark_for_makeup.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
                    draw_landmark_for_makeup.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
                    draw_landmark_for_makeup.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)
                    # Make left and right eyes filled with Green
                    draw_landmark_for_makeup.polygon(face_landmarks['left_eye'], fill=(0, 255, 0, 0))
                    draw_landmark_for_makeup.polygon(face_landmarks['right_eye'], fill=(0, 255, 0, 0))
                    # Eyeliner to left and right eyes as lines
                    draw_landmark_for_makeup.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]],
                                                  fill=(0, 0, 0, 90), width=6)
                    draw_landmark_for_makeup.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]],
                                                  fill=(0, 0, 0, 90), width=6)
                # 显示图像
                img = cv2.cvtColor(numpy.asarray(pil_image), cv2.COLOR_RGB2BGR)
                self.__outImageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.__drawImage(self.outImageView, self.__outImageRGB)
            except:
                QMessageBox.information(self, '提示', '画面中人脸特征不够明显！')
        else:
            QMessageBox.information(self, '提示', '您还未读入图像！')
            return

    # -----------------------------------关于-----------------------------------
    # 关于作者
    def __aboutAuthor(self):
        QMessageBox.information(None, '关于作者', '数字图像处理软件2.0\n\nCopyright © 2022–2099 数媒2002 李子睿\n\n保留一切权利')
    # 学习侦测

    def push_val(self,val):
        if val < 800:
            if len(eyeq) <= eyeq_len:
                eyeq.append(val)
            else:
                eyeq.append(val)
                eyeq.pop(0)
        return self.avg_eyeq()
    def avg_eyeq(self):
        # calculate average
        avg = 0
        for i in eyeq:
            avg = avg + i
        avg = avg / (len(eyeq) + 1)
        return avg
    def detect_and_draw(self,img, gray):
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[int((y + h / 4)):int((y + 0.55 * h)), int((x + 0.13 * w)):int((x + w - 0.13 * w))]
            roi_color = img[int((y + h / 4)):int((y + 0.55 * h)), int((x + 0.13 * w)):int((x + w - 0.13 * w))]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            max_eyes = 2
            cnt_eye = 0
            for (ex, ey, ew, eh) in eyes:
                if (cnt_eye == max_eyes):
                    break
                image_name = 'Eye_' + str(cnt_eye)

                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
                roi_eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                roi_eye_color = roi_color[ey:ey + eh, ex:ex + ew]
                # create & normalize histogram ---------
                hist = cv2.calcHist([roi_eye_gray], [0], None, [256], [0, 256])
                histn = []
                max_val = 0
                for i in hist:
                    value = int(i[0])
                    histn.append(value)
                    if value > max_val:
                        max_val = value
                for index, value in enumerate(histn):
                    histn[index] = ((value * 256) / max_val)
                threshold = numpy.argmax(histn)
                roi_eye_gray2 = roi_eye_gray.copy()
                total_white = 0
                total_black = 0
                for i in range(0, roi_eye_gray2.shape[0]):
                    for j in range(0, roi_eye_gray2.shape[1]):
                        pixel_value = roi_eye_gray2[i, j]
                        if pixel_value >= threshold:
                            roi_eye_gray2[i, j] = 255
                            total_white = total_white + 1
                        else:
                            roi_eye_gray2[i, j] = 0
                            total_black = total_black + 1
                binary = cv2.resize(roi_eye_gray2, None, fx=2, fy=2)
                # cv2.imshow('binary', binary)
                if image_name == "Eye_0":
                    ag = self.push_val(total_white)
                if simulate_real_time == "true":
                    pass
                    # put number on image
                    if (cnt_eye == 0):
                        cv2.putText(img, "" + str(total_white), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                    else:
                        cv2.putText(img, "" + str(total_white), (520, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                    cv2.putText(img, "" + str(threshold), (10, 240), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                else:
                    # Plot Histogram
                    plt.subplot(2, 3, ((cnt_eye * 3) + 1)), plt.hist(roi_eye_gray.ravel(), 256, [0, 256])
                    plt.title(image_name + ' Hist')
                    # Plot Eye Images
                    plt.subplot(2, 3, ((cnt_eye * 3) + 2)), plt.imshow(roi_eye_color, 'gray')
                    plt.title(image_name + ' Image Threshold')
                    # Plot Eye Images after threshold
                    plt.subplot(2, 3, ((cnt_eye * 3) + 3)), plt.imshow(roi_eye_gray2, 'gray')
                    plt.title(image_name + ' Image')
                cnt_eye = cnt_eye + 1
            if len(eyes) == 0:
                ag = self.push_val(0)
            # Decision Making
            average = self.avg_eyeq()
            if average > 30:
                pass
            else:
                winsound.Beep(1000, 100)
        cv2.imshow('frame', img)
    def __kaishi(self):
        if simulate_real_time == "true":
            global cap, frame
            cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            while True:
                # time.sleep(1)
                ret, frame = cap.read()
                if frame.any() is not None:
                    cv2.imshow('frame', frame)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.detect_and_draw(frame, gray)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                else:
                    print("Frame Grabbed Problem")
                    continue

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
