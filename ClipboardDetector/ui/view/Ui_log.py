# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\HP\Desktop\NIS3366-ClipboardDetector\ClipboardDetector\ui\resource\ui\log.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_log(object):
    def setupUi(self, log):
        log.setObjectName("log")
        log.resize(943, 571)
        self.tableWidget = TableWidget(log)
        self.tableWidget.setGeometry(QtCore.QRect(20, 40, 901, 511))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)

        self.retranslateUi(log)
        QtCore.QMetaObject.connectSlotsByName(log)

    def retranslateUi(self, log):
        _translate = QtCore.QCoreApplication.translate
        log.setWindowTitle(_translate("log", "Form"))
from qfluentwidgets import TableWidget
