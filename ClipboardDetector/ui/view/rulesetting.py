# coding:utf-8
import sqlite3
import psutil
from PyQt5.QtWidgets import (
    QWidget,
    QFileDialog,
    QDialog,
    QVBoxLayout,
    QListWidget,
    QPushButton,
    QAbstractItemView,
    QTableView,
    QHeaderView,
    QLineEdit,
    QHBoxLayout,
    QLabel,
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QSortFilterProxyModel

from qfluentwidgets import FluentIcon, InfoBar, InfoBarPosition

from view.Ui_rulesetting import Ui_rulesetting


class ProcessTableModel(QAbstractTableModel):
    """进程TableModel"""

    def __init__(self, process_list):
        super().__init__()
        self.process_list = process_list
        self.header = ["进程名称", "PID"]

    def rowCount(self, parent=None):
        return len(self.process_list)

    def columnCount(self, parent=None):
        return len(self.header)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            row = index.row()
            col = index.column()
            process = self.process_list[row]
            if col == 0:
                return process["name"]
            elif col == 1:
                return str(process["pid"])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.header[section]
        return None


class RuleSetting(Ui_rulesetting, QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)

        items = [
            "深度模型",
            "深度模型onnx加速",
            "深度模型+正则匹配（推荐）",
            "低强度正则匹配",
            "中等强度正则匹配",
            "高强度正则匹配",
        ]
        self.pushButton.addItems(items)
        self.pushButton.setPlaceholderText("请选择使用的检测器")
        self.pushButton.setCurrentIndex(-1)

        self.pushButton.currentIndexChanged.connect(self.on_index_changed)
        self.pushButton_2.clicked.connect(self.choosewhitesheetapp)  # 连接按钮和槽函数
        self.pushButton_3.clicked.connect(self.remove_whitelist_app)

        self.load_whitelist()  # 初始化时加载白名单

    def on_index_changed(self, index):
        if self.pushButton.itemText(index) == "深度模型":
            InfoBar.success(
                title="选择深度模型",
                content="请等待模型加载完成，深度学习模型比正则匹配更准确，但是速度较慢",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )
        elif self.pushButton.itemText(index) == "深度模型onnx加速":
            InfoBar.success(
                title="选择深度模型onnx加速",
                content="此选项更利于配置较低的设备，能在不影响精度的情况下提高检测速度",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )
        elif self.pushButton.itemText(index) == "深度模型+正则匹配（推荐）":
            InfoBar.success(
                title="选择深度模型+正则匹配",
                content="此选项利用深度模型和正则匹配相结合，能够在不影响速度的情况下提高检测精度",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )
        elif self.pushButton.itemText(index) == "低强度正则匹配":
            InfoBar.success(
                title="选择低强度正则匹配",
                content="此选项利用正则匹配，速度快，但是精度较低，易产生误报",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )
        elif self.pushButton.itemText(index) == "中等强度正则匹配":
            InfoBar.success(
                title="选择中等强度正则匹配",
                content="此选项利用正则匹配，精度较高，适合大多数用户",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )
        elif self.pushButton.itemText(index) == "高强度正则匹配":
            InfoBar.success(
                title="选择高弪度正则匹配",
                content="此选项利用正则匹配，精度最高，但是易产生漏报",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )
        else:
            InfoBar.error(
                title="未选择检测器",
                content="请先选择使用的检测器",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )

    def choosewhitesheetapp(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("选择要添加到白名单的程序")
        dialog.resize(800, 600)  # 调整窗口大小
        layout = QVBoxLayout(dialog)

        # 1. 搜索框
        search_layout = QHBoxLayout()
        search_label = QLabel("搜索:")  # Import QLabel
        search_input = QLineEdit()
        search_layout.addWidget(search_label)
        search_layout.addWidget(search_input)
        layout.addLayout(search_layout)

        # 2. 进程列表
        process_list = []
        for proc in psutil.process_iter(["pid", "name"]):
            process_list.append(proc.info)

        self.table_model = ProcessTableModel(process_list)
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.table_model)

        table_view = QTableView()
        table_view.setModel(self.proxy_model)
        table_view.setSelectionBehavior(QTableView.SelectRows)  # 选择整行
        table_view.setSelectionMode(QTableView.MultiSelection)  # 允许多选
        table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )  # 自动调整列宽
        layout.addWidget(table_view)

        # 3. 确定按钮
        ok_button = QPushButton("确定", dialog)
        layout.addWidget(ok_button)

        # 4. 搜索功能
        def filter_process_list(text):
            self.proxy_model.setFilterKeyColumn(0)  # 搜索进程名称列
            self.proxy_model.setFilterFixedString(text)
            self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)  # 忽略大小写

        search_input.textChanged.connect(filter_process_list)

        def on_ok_clicked():
            selected_rows = table_view.selectionModel().selectedRows()
            for row in selected_rows:
                index = self.proxy_model.mapToSource(row)
                process_name = self.table_model.process_list[index.row()]["name"]
                self.add_to_whitelist(process_name)
            self.load_whitelist()
            dialog.accept()

        ok_button.clicked.connect(on_ok_clicked)

        # 5. 样式表
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #E1F5FE; /* 淡蓝色 */
            }
            QLabel {
                font-size: 14px;
                color: #333333;
            }
            QLineEdit {
                border: 1px solid #81D4FA; /* 浅蓝色边框 */
                padding: 5px;
                font-size: 14px;
                background-color: #FFFFFF;
            }
            QTableView {
                border: 1px solid #81D4FA; /* 浅蓝色边框 */
                font-size: 14px;
                background-color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #B3E5FC; /* 浅蓝色表头 */
                border: 1px solid #81D4FA; /* 浅蓝色边框 */
                font-size: 14px;
            }
            QPushButton {
                background-color: #4FC3F7; /* 浅蓝色按钮 */
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #29B6F6; /* 更深的浅蓝色 */
            }
        """
        )

        dialog.setLayout(layout)
        dialog.exec_()

    def remove_whitelist_app(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("选择要删除的白名单程序")
        layout = QVBoxLayout(dialog)

        whitelist_list = QListWidget()
        whitelist_list.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(whitelist_list)

        # 从数据库加载白名单程序
        conn = sqlite3.connect("whitelist.db")
        cursor = conn.cursor()
        cursor.execute("SELECT app_name FROM whitelist")
        rows = cursor.fetchall()
        for row in rows:
            whitelist_list.addItem(row[0])
        conn.close()

        ok_button = QPushButton("确定", dialog)
        layout.addWidget(ok_button)

        def on_ok_clicked():
            selected_items = whitelist_list.selectedItems()
            for item in selected_items:
                app_name = item.text()
                self.remove_from_whitelist(app_name)
            self.load_whitelist()
            dialog.accept()

        ok_button.clicked.connect(on_ok_clicked)

        dialog.setLayout(layout)
        dialog.exec_()

    def add_to_whitelist(self, app_name):
        conn = sqlite3.connect("whitelist.db")  # 连接到SQLite数据库
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS whitelist (app_name TEXT)")  # 创建表
        cursor.execute(
            "INSERT INTO whitelist (app_name) VALUES (?)", (app_name,)
        )  # 插入数据
        conn.commit()
        conn.close()

    def remove_from_whitelist(self, app_name):
        conn = sqlite3.connect("whitelist.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM whitelist WHERE app_name=?", (app_name,))
        conn.commit()
        conn.close()

    def load_whitelist(self):
        conn = sqlite3.connect("whitelist.db")
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS whitelist (app_name TEXT)"
        )  # 确保表存在
        cursor.execute("SELECT app_name FROM whitelist")
        rows = cursor.fetchall()
        self.listWidget.clear()  # 清空列表
        for row in rows:
            self.listWidget.addItem(row[0])  # 添加到列表
        conn.close()
