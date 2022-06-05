# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4



from PyQt5 import QtCore, QtGui, QtWidgets

#mainwindow->memu add action（窗口与菜单）->translate UI->（菜单与后台）在 main 类中初始化 trigger connect到某一个具体函数
#用上一步命名的触发器函数->完善具体函数->调试
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QStyle, QApplication


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        style = QApplication.style()
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(896, 577)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.outImageView = QtWidgets.QGraphicsView(self.centralwidget)
        self.outImageView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.outImageView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.outImageView.setObjectName("outImageView")
        self.gridLayout.addWidget(self.outImageView, 1, 1, 1, 1)
        self.srcImageView = QtWidgets.QGraphicsView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.srcImageView.sizePolicy().hasHeightForWidth())
        self.srcImageView.setSizePolicy(sizePolicy)
        self.srcImageView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.srcImageView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.srcImageView.setObjectName("srcImageView")
        self.gridLayout.addWidget(self.srcImageView, 1, 0, 1, 1)
        self.srcImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.srcImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.srcImageLabel.setObjectName("srcImageLabel")
        self.gridLayout.addWidget(self.srcImageLabel, 0, 0, 1, 1)
        self.outImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.outImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.outImageLabel.setObjectName("outImageLabel")
        self.gridLayout.addWidget(self.outImageLabel, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 896, 24))
        self.menubar.setObjectName("menubar")
        self.fileMenu = QtWidgets.QMenu(self.menubar)
        self.fileMenu.setObjectName("fileMenu")
        self.resetImageMenu = QtWidgets.QMenu(self.menubar)
        self.resetImageMenu.setObjectName("resetImageMenu")
        self.lvjingMenu = QtWidgets.QMenu(self.menubar)
        self.lvjingMenu.setObjectName("lvjingMenu")
        self.aboutMenu = QtWidgets.QMenu(self.menubar)
        self.aboutMenu.setObjectName("aboutMenu")
        self.grayMappingMenu = QtWidgets.QMenu(self.menubar)
        self.grayMappingMenu.setObjectName("grayMappingMenu")
        self.operateImageMenu = QtWidgets.QMenu(self.menubar)
        self.operateImageMenu.setObjectName("operateImageMenu")
        #
        self.fusionMenu = QtWidgets.QMenu(self.operateImageMenu)
        self.fusionMenu.setObjectName("fusionMenu")
        #
        self.histogramMenu = QtWidgets.QMenu(self.menubar)
        self.histogramMenu.setObjectName("histogramMenu")
        self.noiseMenu = QtWidgets.QMenu(self.menubar)
        self.noiseMenu.setObjectName("noiseMenu")
        #以空域滤波三级菜单为例
        #首先定义一级菜单于menubar主栏
        self.filterMenu = QtWidgets.QMenu(self.menubar)
        self.filterMenu.setObjectName("filterMenu")
        #然后定义二级菜单，分为锐化与平滑两个菜单，都依赖在一级菜单下
        self.smoothMenu = QtWidgets.QMenu(self.filterMenu)
        self.smoothMenu.setObjectName("smoothMenu")
        self.sharpMenu = QtWidgets.QMenu(self.filterMenu)
        self.sharpMenu.setObjectName("sharpMenu")
        #之后定义三级菜单，在图像平滑处理的中值滤波中，分为调用opencv库与自设计的模块
        #这个三级菜单依赖于smoothMenu下
        self.middleMenu = QtWidgets.QMenu(self.filterMenu)
        self.middleMenu.setObjectName("middleMenu")
        MainWindow.setMenuBar(self.menubar)
        #刷新菜单模块，进行更新
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.otherMenu = QtWidgets.QMenu(self.menubar)
        self.otherMenu.setObjectName("otherMenu")
        #图像截取菜单
        self.jiequMenu = QtWidgets.QMenu(self.menubar)
        self.jiequMenu.setObjectName("jiequMenu")
        #特征描述菜单
        self.tezhengMenu = QtWidgets.QMenu(self.menubar)
        self.tezhengMenu.setObjectName("tezhengMenu")
#####  以上是对菜单及子菜单的管理，让菜单管理与act实现管理分开，方便修改

#####  定义操作，此时还未与按钮联系
        self.openFileAction = QtWidgets.QAction(MainWindow)
        self.openFileAction.setObjectName("openFileAction")
        self.getFileAction = QtWidgets.QAction(MainWindow)
        self.getFileAction.setObjectName("getFileAction")
        self.saveFileAction = QtWidgets.QAction(MainWindow)
        self.saveFileAction.setObjectName("saveFileAction")
        self.saveFileAsAction = QtWidgets.QAction(MainWindow)
        self.saveFileAsAction.setObjectName("saveFileAsAction")
        self.exitAppAction = QtWidgets.QAction(MainWindow)
        self.exitAppAction.setObjectName("exitAppAction")
        self.resetImageAction = QtWidgets.QAction(MainWindow)
        self.resetImageAction.setObjectName("resetImageAction")
        self.aboutAction = QtWidgets.QAction(MainWindow)
        self.aboutAction.setObjectName("aboutAction")
        self.actiongg_2 = QtWidgets.QAction(MainWindow)
        self.actiongg_2.setObjectName("actiongg_2")
        self.grayAction = QtWidgets.QAction(MainWindow)
        self.grayAction.setObjectName("grayAction")
        self.binaryAction = QtWidgets.QAction(MainWindow)
        self.binaryAction.setObjectName("binaryAction")
        self.reverseAction = QtWidgets.QAction(MainWindow)
        self.reverseAction.setObjectName("reverseAction")
        self.imageAddAction = QtWidgets.QAction(MainWindow)
        self.imageAddAction.setObjectName("imageAddAction")
        self.imageSubtractAction = QtWidgets.QAction(MainWindow)
        self.imageSubtractAction.setObjectName("imageSubtractAction")
        self.imageMultiplyAction = QtWidgets.QAction(MainWindow)
        self.imageMultiplyAction.setObjectName("imageMultiplyAction")
        self.imagefusionAction = QtWidgets.QAction(MainWindow)
        self.imagefusionAction.setObjectName("imagefusionAction")
        self.imagefusionzixuanAction = QtWidgets.QAction(MainWindow)
        self.imagefusionzixuanAction.setObjectName("imagefusionzixuanAction")

        self.histogramAction = QtWidgets.QAction(MainWindow)
        self.histogramAction.setObjectName("histogramAction")
        self.histogramEqAction = QtWidgets.QAction(MainWindow)
        self.histogramEqAction.setObjectName("histogramEqAction")
        self.lightAction = QtWidgets.QAction(MainWindow)
        self.lightAction.setObjectName("lightAction")
        self.contrastAction = QtWidgets.QAction(MainWindow)
        self.contrastAction.setObjectName("contrastAction")
        self.sharpAction = QtWidgets.QAction(MainWindow)
        self.sharpAction.setObjectName("sharpAction")
        self.zoomAction = QtWidgets.QAction(MainWindow)
        self.zoomAction.setObjectName("zoomAction")
        self.rotateAction = QtWidgets.QAction(MainWindow)
        self.rotateAction.setObjectName("rotateAction")
        self.caijianAction = QtWidgets.QAction(MainWindow)
        self.caijianAction.setObjectName("caijianAction")
        self.actiongg = QtWidgets.QAction(MainWindow)
        self.actiongg.setObjectName("actiongg")
        self.saturationAction = QtWidgets.QAction(MainWindow)
        self.saturationAction.setObjectName("saturationAction")
        self.hueAction = QtWidgets.QAction(MainWindow)
        self.hueAction.setObjectName("hueAction")
        self.reColorAction = QtWidgets.QAction(MainWindow)
        self.reColorAction.setObjectName("reColorAction")
        self.addGaussianNoiseAction = QtWidgets.QAction(MainWindow)
        self.addGaussianNoiseAction.setObjectName("addGaussianNoiseAction")
        self.diedaiAction = QtWidgets.QAction(MainWindow)
        self.diedaiAction.setObjectName("diedaiAction")
        self.zhongziAction = QtWidgets.QAction(MainWindow)
        self.zhongziAction.setObjectName("zhongziAction")
        self.renlianAction = QtWidgets.QAction(MainWindow)
        self.renlianAction.setObjectName("renlianAction")
        self.renlianjiequAction = QtWidgets.QAction(MainWindow)
        self.renlianjiequAction.setObjectName("renlianjiequAction")
        self.actiongg_3 = QtWidgets.QAction(MainWindow)
        self.actiongg_3.setObjectName("actiongg_3")
        self.actiongg_4 = QtWidgets.QAction(MainWindow)
        self.actiongg_4.setObjectName("actiongg_4")
        self.meanValueAction = QtWidgets.QAction(MainWindow)
        self.meanValueAction.setObjectName("meanValueAction")
        self.medianValueAction = QtWidgets.QAction(MainWindow)
        self.medianValueAction.setObjectName("medianValueAction")
        self.zishixianmedianValueAction = QtWidgets.QAction(MainWindow)
        self.zishixianmedianValueAction.setObjectName("zishixianmedianValueAction")
        self.guasslvboAction = QtWidgets.QAction(MainWindow)
        self.guasslvboAction.setObjectName("guasslvboAction")

        self.sobelAction = QtWidgets.QAction(MainWindow)
        self.sobelAction.setObjectName("sobelAction")
        self.prewittAction = QtWidgets.QAction(MainWindow)
        self.prewittAction.setObjectName("prewittAction")
        self.laplacianAction = QtWidgets.QAction(MainWindow)
        self.laplacianAction.setObjectName("laplacianAction")
        self.CannyAction = QtWidgets.QAction(MainWindow)
        self.CannyAction.setObjectName("CannyAction")
        self.addUiformNoiseAction = QtWidgets.QAction(MainWindow)
        self.addUiformNoiseAction.setObjectName("addUiformNoiseAction")
        self.addImpulseNoiseAction = QtWidgets.QAction(MainWindow)
        self.addImpulseNoiseAction.setObjectName("addImpulseNoiseAction")
        self.addbosongNoiseAction = QtWidgets.QAction(MainWindow)
        self.addbosongNoiseAction.setObjectName("addbosongNoiseAction")
        self.fileMenu.addAction(self.openFileAction)
        self.fudiaoAction = QtWidgets.QAction(MainWindow)
        self.fudiaoAction.setObjectName("fudiaoAction")
        self.maoboliAction = QtWidgets.QAction(MainWindow)
        self.maoboliAction.setObjectName("maoboliAction")
        self.masaikeAction = QtWidgets.QAction(MainWindow)
        self.masaikeAction.setObjectName("masaikeAction")
        self.katonghuaAction= QtWidgets.QAction(MainWindow)
        self.katonghuaAction.setObjectName("katonghuaAction")
        self.biankuangAction = QtWidgets.QAction(MainWindow)
        self.biankuangAction.setObjectName("biankuangAction")

        self.ronghekuangAction = QtWidgets.QAction(MainWindow)
        self.ronghekuangAction.setObjectName("ronghekuangAction")
        self.pingtuAction = QtWidgets.QAction(MainWindow)
        self.pingtuAction.setObjectName("pingtuAction")

        #相片滤镜
        #怀旧滤镜
        self.huaijiuAction = QtWidgets.QAction(MainWindow)
        self.huaijiuAction.setObjectName("huaijiuAction")
        #光晕滤镜
        self.guangyunAction = QtWidgets.QAction(MainWindow)
        self.guangyunAction.setObjectName("guangyunaction")
        #流年滤镜
        self.liunianAction = QtWidgets.QAction(MainWindow)
        self.liunianAction.setObjectName("liunianAction")
        #人脸一键美化
        self.renlianmakeupAction = QtWidgets.QAction(MainWindow)
        self.renlianmakeupAction.setObjectName("renlianjiemakeupAction")
        #特征描述
        #区域面积
        self.quyumianjiAction = QtWidgets.QAction(MainWindow)
        self.quyumianjiAction.setObjectName("quyumianjiAction")
        #区域周长
        self.quyuzhouchangAction = QtWidgets.QAction(MainWindow)
        self.quyuzhouchangAction.setObjectName("quyuzhouchangAction")
        #最小外接矩形面积
        self.quyujuxingAction = QtWidgets.QAction(MainWindow)
        self.quyujuxingAction.setObjectName("quyujuxingAction")
        #细长度
        self.quyuxichangduAction = QtWidgets.QAction(MainWindow)
        self.quyuxichangduAction.setObjectName("quyuxichangduAction")
        #矩形度（区域占空比）（轮廓区域面积除以最小外接矩形面积）
        self.quyuzhankongbiAction = QtWidgets.QAction(MainWindow)
        self.quyuzhankongbiAction.setObjectName("quyuzhankongbiAction")
        #重心
        self.quyuzhongxinAction = QtWidgets.QAction(MainWindow)
        self.quyuzhongxinAction.setObjectName("quyuzhongxinAction")
        #添加图标标识
        self.openFileAction.setIcon(style.standardIcon(QStyle.SP_DialogOpenButton))
        self.saveFileAction.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
        self.resetImageAction.setIcon(style.standardIcon(QStyle.SP_FileDialogBack))
        self.saveFileAsAction.setIcon(style.standardIcon(QStyle.SP_DirLinkIcon))
        self.exitAppAction.setIcon(style.standardIcon(QStyle.SP_BrowserStop))
        self.fudiaoAction.setIcon(style.standardIcon(QStyle.SP_DialogHelpButton))
        self.maoboliAction.setIcon(style.standardIcon(QStyle.SP_DialogHelpButton))
        self.aboutAction.setIcon(style.standardIcon(QStyle.SP_MessageBoxQuestion))
        self.getFileAction.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        #添加快捷按钮键
        self.openFileAction.setShortcut(Qt.CTRL + Qt.Key_O)
        self.saveFileAction.setShortcut(Qt.CTRL + Qt.Key_S)
        self.resetImageAction.setShortcut(Qt.CTRL + Qt.Key_Z)
        #对菜单添加操作
        self.fileMenu.addAction(self.getFileAction)
        self.fileMenu.addAction(self.saveFileAction)
        self.fileMenu.addAction(self.saveFileAsAction)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAppAction)
        self.resetImageMenu.addAction(self.resetImageAction)
        self.aboutMenu.addAction(self.aboutAction)
        self.grayMappingMenu.addAction(self.grayAction)
        self.grayMappingMenu.addAction(self.binaryAction)
        self.grayMappingMenu.addAction(self.reverseAction)
        self.grayMappingMenu.addSeparator()
        self.grayMappingMenu.addAction(self.lightAction)
        self.grayMappingMenu.addAction(self.contrastAction)
        self.grayMappingMenu.addAction(self.sharpAction)
        self.grayMappingMenu.addAction(self.saturationAction)
        self.grayMappingMenu.addAction(self.hueAction)
        self.operateImageMenu.addAction(self.imageAddAction)
        self.operateImageMenu.addAction(self.imageSubtractAction)
        self.operateImageMenu.addAction(self.imageMultiplyAction)

        self.operateImageMenu.addAction(self.fusionMenu.menuAction())
        self.fusionMenu.addAction(self.imagefusionAction)
        self.fusionMenu.addAction(self.imagefusionzixuanAction)


        self.operateImageMenu.addAction(self.zoomAction)
        self.operateImageMenu.addAction(self.rotateAction)

        self.histogramMenu.addAction(self.histogramAction)
        self.histogramMenu.addAction(self.histogramEqAction)
        #图像截取
        self.jiequMenu.addAction(self.caijianAction)
        self.jiequMenu.addAction(self.diedaiAction)
        self.jiequMenu.addAction(self.zhongziAction)
        self.jiequMenu.addAction(self.renlianAction)
        self.jiequMenu.addAction(self.renlianjiequAction)
        #特征描述
        #区域面积
        self.tezhengMenu.addAction(self.quyumianjiAction)
        #区域周长
        self.tezhengMenu.addAction(self.quyuzhouchangAction)
        #最小外接矩形
        self.tezhengMenu.addAction(self.quyujuxingAction)
        #矩形度
        self.tezhengMenu.addAction(self.quyuzhankongbiAction)
        #细长度
        self.tezhengMenu.addAction(self.quyuxichangduAction)
        #重心
        self.tezhengMenu.addAction(self.quyuzhongxinAction)

        self.noiseMenu.addAction(self.addGaussianNoiseAction)
        self.noiseMenu.addAction(self.addUiformNoiseAction)
        self.noiseMenu.addAction(self.addImpulseNoiseAction)
        self.noiseMenu.addAction(self.addbosongNoiseAction)
        self.smoothMenu.addAction(self.meanValueAction)
        self.middleMenu.addAction(self.medianValueAction)
        self.middleMenu.addAction(self.zishixianmedianValueAction)
        self.smoothMenu.addAction(self.guasslvboAction)
        self.sharpMenu.addAction(self.sobelAction)
        self.sharpMenu.addAction(self.prewittAction)
        self.sharpMenu.addAction(self.laplacianAction)
        self.sharpMenu.addAction(self.CannyAction)
        self.filterMenu.addAction(self.smoothMenu.menuAction())
        self.filterMenu.addAction(self.sharpMenu.menuAction())
        self.smoothMenu.addAction(self.middleMenu.menuAction())
        self.otherMenu.addAction(self.fudiaoAction)
        self.otherMenu.addAction(self.maoboliAction)
        self.otherMenu.addAction(self.masaikeAction)
        self.otherMenu.addAction(self.katonghuaAction)

        self.otherMenu.addSeparator()
        self.otherMenu.addAction(self.biankuangAction)
        self.otherMenu.addAction(self.ronghekuangAction)
        self.otherMenu.addAction(self.pingtuAction)
        #滤镜菜单
        self.lvjingMenu.addAction(self.huaijiuAction)
        self.lvjingMenu.addAction(self.guangyunAction)
        self.lvjingMenu.addAction(self.liunianAction)
        self.lvjingMenu.addAction(self.renlianmakeupAction)

        self.menubar.addAction(self.fileMenu.menuAction())
        self.menubar.addAction(self.resetImageMenu.menuAction())
        self.menubar.addAction(self.grayMappingMenu.menuAction())
        self.menubar.addAction(self.operateImageMenu.menuAction())
        self.menubar.addAction(self.histogramMenu.menuAction())
        self.menubar.addAction(self.tezhengMenu.menuAction())
        self.menubar.addAction(self.jiequMenu.menuAction())
        self.menubar.addAction(self.noiseMenu.menuAction())
        self.menubar.addAction(self.filterMenu.menuAction())
        self.menubar.addAction(self.otherMenu.menuAction())
        self.menubar.addAction(self.lvjingMenu.menuAction())
        self.menubar.addAction(self.aboutMenu.menuAction())
        # self.menubar.addAction()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    #将英文元素全部于translate部分重置为中文，方便统一管理
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图像处理软件"))
        self.srcImageLabel.setText(_translate("MainWindow", "原图预览"))
        self.outImageLabel.setText(_translate("MainWindow", "处理后的图片预览"))
        self.fileMenu.setTitle(_translate("MainWindow", "文件"))
        self.resetImageMenu.setTitle(_translate("MainWindow", "重置"))
        self.aboutMenu.setTitle(_translate("MainWindow", "关于"))
        self.grayMappingMenu.setTitle(_translate("MainWindow", "图像预处理与增强"))
        self.operateImageMenu.setTitle(_translate("MainWindow", "图像运算"))
        self.histogramMenu.setTitle(_translate("MainWindow", "直方图均衡"))
        self.noiseMenu.setTitle(_translate("MainWindow", "噪声"))
        self.filterMenu.setTitle(_translate("MainWindow", "空域滤波"))
        self.middleMenu.setTitle(_translate("MainWindow", "中值滤波"))
        self.smoothMenu.setTitle(_translate("MainWindow", "平滑滤波"))
        self.fusionMenu.setTitle(_translate("MainWindow", "融合"))
        self.sharpMenu.setTitle(_translate("MainWindow", "锐化滤波"))
        self.jiequMenu.setTitle(_translate("MainWindow", "图像截取"))
        self.tezhengMenu.setTitle(_translate("MainWindow", "特征描述"))
        self.quyumianjiAction.setText(_translate("MainWindow", "面积"))
        self.quyuzhouchangAction.setText(_translate("MainWindow", "周长"))
        self.lvjingMenu.setTitle(_translate("MainWindow", "相片滤镜与美化"))
        self.openFileAction.setText(_translate("MainWindow", "打开"))
        self.saveFileAction.setText(_translate("MainWindow", "保存"))
        self.saveFileAsAction.setText(_translate("MainWindow", "另存为"))
        self.exitAppAction.setText(_translate("MainWindow", "退出"))
        self.resetImageAction.setText(_translate("MainWindow", "恢复到原始图片"))
        self.aboutAction.setText(_translate("MainWindow", "关于作者"))
        self.actiongg_2.setText(_translate("MainWindow", "gg"))
        self.grayAction.setText(_translate("MainWindow", "灰度化"))
        self.binaryAction.setText(_translate("MainWindow", "二值化"))
        self.reverseAction.setText(_translate("MainWindow", "颜色反转"))
        self.imageAddAction.setText(_translate("MainWindow", "加"))
        self.imageSubtractAction.setText(_translate("MainWindow", "减"))
        self.imageMultiplyAction.setText(_translate("MainWindow", "乘"))
        self.imagefusionAction.setText(_translate("MainWindow", "1:1融合"))
        self.imagefusionzixuanAction.setText(_translate("MainWindow", "自定义融合"))
        self.histogramAction.setText(_translate("MainWindow", "归一化直方图"))
        self.histogramEqAction.setText(_translate("MainWindow", "直方图均衡化"))
        self.lightAction.setText(_translate("MainWindow", "亮度"))
        self.contrastAction.setText(_translate("MainWindow", "对比度"))
        self.sharpAction.setText(_translate("MainWindow", "锐度"))
        self.zoomAction.setText(_translate("MainWindow", "缩放"))
        self.rotateAction.setText(_translate("MainWindow", "旋转"))
        self.actiongg.setText(_translate("MainWindow", "gg"))
        self.saturationAction.setText(_translate("MainWindow", "饱和度"))
        self.hueAction.setText(_translate("MainWindow", "色调"))
        self.reColorAction.setText(_translate("MainWindow", "重新着色"))
        self.addGaussianNoiseAction.setText(_translate("MainWindow", "加高斯噪声"))
        self.addbosongNoiseAction.setText(_translate("MainWindow", "加泊松噪声"))
        self.actiongg_3.setText(_translate("MainWindow", "gg"))
        self.actiongg_4.setText(_translate("MainWindow", "gg"))
        self.meanValueAction.setText(_translate("MainWindow", "均值滤波"))
        self.guasslvboAction.setText(_translate("MainWindow", "高斯滤波"))
        self.medianValueAction.setText(_translate("MainWindow", "opencv中值滤波"))
        self.zishixianmedianValueAction.setText(_translate("MainWindow", "自实现中值滤波"))
        self.sobelAction.setText(_translate("MainWindow", "Sobel算子"))
        self.prewittAction.setText(_translate("MainWindow", "Prewitt算子"))
        self.laplacianAction.setText(_translate("MainWindow", "拉普拉斯算子"))
        self.CannyAction.setText(_translate("MainWindow", "Canny算子"))
        self.addUiformNoiseAction.setText(_translate("MainWindow", "加椒盐噪声"))
        self.addImpulseNoiseAction.setText(_translate("MainWindow", "加随机噪声"))
        self.otherMenu.setTitle(_translate("MainWindow", "图片美化"))
        self.fudiaoAction.setText(_translate("MainWindow", "浮雕效果"))
        self.maoboliAction.setText(_translate("MainWindow", "毛玻璃效果"))
        self.masaikeAction.setText(_translate("MainWindow", "马赛克效果"))
        self.caijianAction.setText(_translate("MainWindow", "图像裁剪"))
        self.getFileAction.setText(_translate("MainWindow", "摄像头捕捉图片"))
        self.diedaiAction.setText(_translate("MainWindow", "迭代阈值分割"))
        self.zhongziAction.setText(_translate("MainWindow", "区域增长"))
        self.biankuangAction.setText(_translate("MainWindow", "图像加框"))
        self.renlianAction.setText(_translate("MainWindow", "人脸框取"))
        self.renlianjiequAction.setText(_translate("MainWindow", "人脸轮廓勾勒"))
        self.pingtuAction.setText(_translate("MainWindow", "图像拼接"))
        self.ronghekuangAction.setText(_translate("MainWindow", "图像融合加框"))
        self.katonghuaAction.setText(_translate("MainWindow", "图像卡通化"))
        self.huaijiuAction.setText(_translate("MainWindow", "怀旧"))
        self.guangyunAction.setText(_translate("MainWindow", "光晕"))
        self.liunianAction.setText(_translate("MainWindow", "流年"))
        self.renlianmakeupAction.setText(_translate("MainWindow", "人脸一键美化化妆"))
        self.quyujuxingAction.setText(_translate("MainWindow", "最小外接矩形面积"))
        self.quyuzhankongbiAction.setText(_translate("MainWindow", "矩形度"))
        self.quyuxichangduAction.setText(_translate("MainWindow", "细长度"))
        self.quyuzhongxinAction.setText(_translate("MainWindow", "重心"))
