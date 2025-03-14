# coding:utf-8
import sqlite3
import os
from datetime import datetime
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QGraphicsDropShadowEffect, QTableWidgetItem
from PyQt5.QtCore import QTimer
from qfluentwidgets import FluentIcon, setFont, InfoBarIcon

from view.Ui_log import Ui_log


class LogInterface(Ui_log, QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)

        # 数据库路径
        self.db_path = "clipboard.db"

        # 启用边框并设置圆角
        self.tableWidget.setBorderVisible(True)
        self.tableWidget.setBorderRadius(8)
        self.tableWidget.setWordWrap(False)

        # 设置表格列 - 增加到5列，添加来源进程列
        self.tableWidget.setColumnCount(5)

        # 设置水平表头并隐藏垂直表头 - 添加来源进程
        self.tableWidget.setHorizontalHeaderLabels(
            ["时间", "风险等级", "内容哈希值", "来源进程", "预览"]
        )

        # 设置列宽 - 调整以适应新列
        self.tableWidget.setColumnWidth(0, 160)  # 时间列
        self.tableWidget.setColumnWidth(1, 100)  # 风险等级列
        self.tableWidget.setColumnWidth(2, 100)  # 哈希值列
        self.tableWidget.setColumnWidth(3, 120)  # 来源进程列
        self.tableWidget.setColumnWidth(4, 180)  # 预览列

        # 加载数据
        self.load_clipboard_data()

        # 设置定时刷新 (每5秒刷新一次)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.load_clipboard_data)
        self.timer.start(5000)

    def load_clipboard_data(self):
        """从数据库加载剪贴板日志数据"""
        try:
            if not os.path.exists(self.db_path):
                print(f"数据库文件不存在: {self.db_path}")
                return

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 获取最近的100条记录 - 添加process_name列
            cursor.execute(
                """
                SELECT id, timestamp, risk_level, content_hash, process_name, preview 
                FROM clipboard_logs
                ORDER BY timestamp DESC
                LIMIT 100
            """
            )

            logs = cursor.fetchall()
            self.tableWidget.setRowCount(len(logs))

            # 设置风险级别对应的颜色
            risk_colors = {
                "low": QColor(0, 170, 0),  # 绿色
                "medium": QColor(255, 170, 0),  # 橙色
                "high": QColor(255, 0, 0),  # 红色
            }

            for i, log in enumerate(logs):
                # 格式化时间戳以便更好地显示
                try:
                    timestamp = datetime.fromisoformat(log[1])
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_time = str(log[1])

                # 设置各列数据
                time_item = QTableWidgetItem(formatted_time)
                risk_item = QTableWidgetItem(log[2])
                hash_item = QTableWidgetItem(
                    log[3][:15] + "..." if len(log[3]) > 15 else log[3]
                )
                # 添加进程名列
                process_name = log[4] if log[4] else "未知"
                process_item = QTableWidgetItem(process_name)

                preview_item = QTableWidgetItem(
                    log[5][:40] + "..." if len(log[5]) > 40 else log[5]
                )

                # 为风险等级设置颜色
                if log[2] in risk_colors:
                    risk_item.setForeground(risk_colors[log[2]])

                # 修改设置项的顺序，添加进程名项
                self.tableWidget.setItem(i, 0, time_item)
                self.tableWidget.setItem(i, 1, risk_item)
                self.tableWidget.setItem(i, 2, hash_item)
                self.tableWidget.setItem(i, 3, process_item)
                self.tableWidget.setItem(i, 4, preview_item)

            conn.close()

        except Exception as e:
            print(f"加载剪贴板数据失败: {str(e)}")

    def setShadowEffect(self, card: QWidget):
        shadowEffect = QGraphicsDropShadowEffect(self)
        shadowEffect.setColor(QColor(0, 0, 0, 15))
        shadowEffect.setBlurRadius(10)
        shadowEffect.setOffset(0, 0)
        card.setGraphicsEffect(shadowEffect)
