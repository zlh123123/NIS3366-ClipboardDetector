# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\HP\Desktop\NIS3366-ClipboardDetector\ClipboardDetector\ui\resource\ui\rulesetting.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_rulesetting(object):
    def setupUi(self, rulesetting):
        rulesetting.setObjectName("rulesetting")
        rulesetting.resize(749, 571)
        self.frame = CardWidget(rulesetting)
        self.frame.setGeometry(QtCore.QRect(30, 70, 691, 61))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.pushButton = ComboBox(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(20, 20, 171, 28))
        self.pushButton.setObjectName("pushButton")
        self.frame_2 = CardWidget(rulesetting)
        self.frame_2.setGeometry(QtCore.QRect(30, 160, 691, 411))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.listWidget = ListWidget(self.frame_2)
        self.listWidget.setGeometry(QtCore.QRect(20, 50, 651, 361))
        self.listWidget.setObjectName("listWidget")
        self.pushButton_2 = PushButton(self.frame_2)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 10, 111, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = PushButton(self.frame_2)
        self.pushButton_3.setGeometry(QtCore.QRect(160, 10, 111, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label = StrongBodyLabel(rulesetting)
        self.label.setGeometry(QtCore.QRect(30, 50, 91, 16))
        self.label.setObjectName("label")
        self.label_2 = StrongBodyLabel(rulesetting)
        self.label_2.setGeometry(QtCore.QRect(30, 140, 91, 16))
        self.label_2.setObjectName("label_2")

        self.retranslateUi(rulesetting)
        QtCore.QMetaObject.connectSlotsByName(rulesetting)

    def retranslateUi(self, rulesetting):
        _translate = QtCore.QCoreApplication.translate
        rulesetting.setWindowTitle(_translate("rulesetting", "Form"))
        self.pushButton.setText(_translate("rulesetting", "请选择使用的检测器"))
        self.pushButton_2.setText(_translate("rulesetting", "请选择应用"))
        self.pushButton_3.setText(_translate("rulesetting", "删除应用"))
        self.label.setText(_translate("rulesetting", "检测器选择"))
        self.label_2.setText(_translate("rulesetting", "应用白名单"))
from qfluentwidgets import CardWidget, ComboBox, ListWidget, PushButton, StrongBodyLabel
