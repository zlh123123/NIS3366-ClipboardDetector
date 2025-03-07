# coding:utf-8
import sys

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon, QDesktopServices
from PyQt5.QtWidgets import QApplication
from qfluentwidgets import (NavigationItemPosition, MessageBox, setTheme, Theme,
                            NavigationAvatarWidget,  SplitFluentWindow, FluentTranslator)
from qfluentwidgets import FluentIcon as FIF


from view.rulesetting import RuleSetting
from view.loginterface import LogInterface
from view.dashboard import DashBoard
import os


class Window(SplitFluentWindow):

    def __init__(self):
        super().__init__()

        # create sub interface
        self.logInterface = LogInterface(self)
        self.rulesetting = RuleSetting(self)
        self.dashboard = DashBoard(self)

        self.initNavigation()
        self.initWindow()

    def initNavigation(self):
        # add sub interface
        self.addSubInterface(self.logInterface, FIF.VPN, "日志")
        self.addSubInterface(self.rulesetting, FIF.SETTING, "规则设置")
        self.addSubInterface(self.dashboard, FIF.SPEED_HIGH, "数据概览")

        self.navigationInterface.setExpandWidth(280)

    def initWindow(self):
        self.resize(800, 600)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "image", "icon.png")
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle('剪切板安全检测器')

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # setTheme(Theme.DARK)

    app = QApplication(sys.argv)

    # install translator
    translator = FluentTranslator()
    app.installTranslator(translator)

    w = Window()
    w.show()
    app.exec_()
