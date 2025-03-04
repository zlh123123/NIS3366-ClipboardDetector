# coding:utf-8
import sys

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon, QDesktopServices
from PyQt5.QtWidgets import QApplication
from qfluentwidgets import (NavigationItemPosition, MessageBox, setTheme, Theme,
                            NavigationAvatarWidget,  SplitFluentWindow, FluentTranslator)
from qfluentwidgets import FluentIcon as FIF


from view.stop_watch_interface import StopWatchInterface
from view.loginterface import LogInterface


class Window(SplitFluentWindow):

    def __init__(self):
        super().__init__()

        # create sub interface
        self.logInterface = LogInterface(self)
        self.stopWatchInterface = StopWatchInterface(self)

        self.initNavigation()
        self.initWindow()

    def initNavigation(self):
        # add sub interface
        self.addSubInterface(self.logInterface, FIF.RINGER, "日志")
        self.addSubInterface(self.stopWatchInterface, FIF.STOP_WATCH, '秒表')



        self.navigationInterface.setExpandWidth(280)

    def initWindow(self):
        self.resize(950, 560)
        self.setWindowIcon(QIcon('./image/icon.png'))
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
