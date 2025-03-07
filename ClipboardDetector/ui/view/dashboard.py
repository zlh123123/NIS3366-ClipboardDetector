import sqlite3
import os
from datetime import datetime, timedelta
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QWidget, QGraphicsDropShadowEffect, QTableWidgetItem
from PyQt5.QtCore import Qt, QSize

from view.Ui_dashboard import Ui_dashboard
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import io


class DashBoard(Ui_dashboard, QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.image_files = []  # 用于存储生成的图片文件路径
        # 中文乱码问题
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        self.charts_generated = False  # 添加一个标志位
        self.temp_dir = os.path.join(
            "ClipboardDetector", "ui", "resource", "images", "temp"
        )  # 设置临时文件夹路径
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.load_data_and_generate_charts()
        self.listWidget.currentIndexChanged.connect(self.update_label_text)
        self.listWidget.setBorderRadius(15)
        self.listWidget.setItemSize(QSize(600, 431))
        self.listWidget.setFixedSize(QSize(600, 431))
        self.listWidget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)

    def load_data_and_generate_charts(self):
        """加载数据并生成图表"""
        if self.charts_generated:
            return  # 如果已经生成过图表，则直接返回

        conn = sqlite3.connect("clipboard.db")
        df = pd.read_sql_query("SELECT * FROM clipboard_logs", conn)
        # print(df)
        conn.close()

        # 统一图表大小
        fig_width = 8
        fig_height = 6

        # 1. 实时风险仪表盘
        risk_dashboard_image = self.generate_risk_dashboard(df, fig_width, fig_height)
        self.image_files.append(risk_dashboard_image)

        # 3. 敏感数据分布分析饼图
        sensitive_data_pie_image = self.generate_sensitive_data_pie(
            df, fig_width, fig_height
        )
        self.image_files.append(sensitive_data_pie_image)

        # 4. 词云
        wordcloud_image = self.generate_wordcloud(df, fig_width, fig_height)
        self.image_files.append(wordcloud_image)

        # 5. 用户行为时间线
        user_behavior_timeline_image = self.generate_user_behavior_timeline(
            df, fig_width, fig_height
        )
        self.image_files.append(user_behavior_timeline_image)

        # 6. 日历热力图
        calendar_heatmap_image = self.generate_calendar_heatmap(
            df, fig_width, fig_height
        )
        self.image_files.append(calendar_heatmap_image)

        self.add_images_to_flip_view(self.image_files)
        self.charts_generated = True  # 设置标志位为 True

    def generate_risk_dashboard(self, df, fig_width, fig_height):
        """生成实时风险仪表盘"""
        # 提取当日数据
        today = datetime.now().strftime("%Y-%m-%d")
        df_today = df[df["timestamp"].str.contains(today)]

        # 计算风险等级占比
        risk_counts = df_today["risk_level"].value_counts(normalize=True) * 100

        # 创建饼图
        plt.figure(figsize=(fig_width, fig_height))  # 统一图表大小
        colors = sns.color_palette("pastel")  # 使用 seaborn 调色板
        plt.pie(
            risk_counts,
            labels=risk_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            textprops={"fontsize": 12},  # 调整字体大小
        )
        plt.title("当日敏感操作占比", fontsize=16)  # 调整标题字体大小
        plt.gca().set_aspect("equal")  # 使饼图为正圆形

        # 保存图片到临时文件
        img_filename = os.path.join(self.temp_dir, "risk_dashboard.png")
        plt.savefig(img_filename, format="png")
        plt.close()
        return img_filename

    def generate_sensitive_data_pie(self, df, fig_width, fig_height):
        """生成敏感数据分布分析饼图"""
        # 假设检测结果中包含敏感数据类型信息，这里简化为随机生成
        data_types = ["银行卡", "密码", "身份证", "手机号"]
        data = [25, 30, 15, 30]

        # 创建饼图
        plt.figure(figsize=(fig_width, fig_height))  # 统一图表大小
        colors = sns.color_palette("Set2")  # 使用 seaborn 调色板
        plt.pie(
            data, labels=data_types, autopct="%1.1f%%", startangle=140, colors=colors
        )
        plt.title("各类型敏感数据占比", fontsize=16)  # 调整标题字体大小
        plt.gca().set_aspect("equal")  # 使饼图为正圆形

        # 保存图片到临时文件
        img_filename = os.path.join(self.temp_dir, "sensitive_data_pie.png")
        plt.savefig(img_filename, format="png")
        plt.close()
        return img_filename

    def generate_wordcloud(self, df, fig_width, fig_height):
        """生成词云"""
        # 将所有剪贴板内容合并为一个字符串
        text = " ".join(df["preview"].dropna().astype(str).tolist())

        # 创建词云对象
        wordcloud = WordCloud(
            font_path="simhei.ttf",  # 设置字体，以支持中文
            background_color="white",
            width=800,
            height=400,
        ).generate(text)

        # 显示词云
        plt.figure(figsize=(fig_width, fig_height))  # 统一图表大小
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("高频敏感关键词", fontsize=16)  # 调整标题字体大小

        # 保存图片到临时文件
        img_filename = os.path.join(self.temp_dir, "wordcloud.png")
        plt.savefig(img_filename, format="png")
        plt.close()
        return img_filename

    def generate_user_behavior_timeline(self, df, fig_width, fig_height):
        """生成用户行为时间线"""
        # 将时间戳转换为 datetime 对象
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 按小时统计剪贴板操作次数
        hourly_counts = df.groupby(df["timestamp"].dt.hour).size()

        # 创建折线图
        plt.figure(figsize=(fig_width, fig_height))  # 统一图表大小
        plt.plot(
            hourly_counts.index, hourly_counts.values, marker="o", linestyle="-"
        )  # 添加线条
        plt.title("按小时统计的剪贴板操作次数", fontsize=16)  # 调整标题字体大小
        plt.xlabel("小时", fontsize=12)  # 调整轴标签字体大小
        plt.ylabel("操作次数", fontsize=12)  # 调整轴标签字体大小
        plt.xticks(range(24))
        plt.grid(True)

        # 保存图片到临时文件
        img_filename = os.path.join(self.temp_dir, "user_behavior_timeline.png")
        plt.savefig(img_filename, format="png")
        plt.close()
        return img_filename

    def generate_calendar_heatmap(self, df, fig_width, fig_height):
        """生成日历热力图"""
        # 将时间戳转换为 datetime 对象，并设置为索引
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # 按天统计风险事件总数
        daily_risk_counts = df["risk_level"].resample("D").count()

        # 创建一个包含所有日期的 DataFrame
        start_date = daily_risk_counts.index.min().date()
        end_date = daily_risk_counts.index.max().date()
        all_dates = pd.date_range(start_date, end_date)
        all_dates_df = pd.DataFrame(index=all_dates)

        # 将统计数据合并到包含所有日期的 DataFrame 中
        daily_risk_counts = all_dates_df.join(daily_risk_counts, how="left").fillna(0)
        daily_risk_counts.rename(columns={"risk_level": "count"}, inplace=True)

        # 创建日历热力图
        plt.figure(figsize=(fig_width, fig_height))  # 统一图表大小
        sns.heatmap(
            daily_risk_counts.T,
            cmap="YlOrRd",
            cbar_kws={
                "label": "风险事件总数",
                "orientation": "horizontal",
            },  # 水平显示 colorbar
        )
        plt.title("每日风险事件总数", fontsize=16)  # 调整标题字体大小
        plt.xlabel("日期", fontsize=12)  # 调整轴标签字体大小
        plt.ylabel("")
        plt.yticks([])  # 隐藏 y 轴刻度

        # 保存图片到临时文件
        img_filename = os.path.join(self.temp_dir, "calendar_heatmap.png")
        plt.savefig(img_filename, format="png")
        plt.close()
        return img_filename

    def add_images_to_flip_view(self, image_paths):
        """将图片添加到 FlipView"""
        self.listWidget.addImages(image_paths)

    def update_label_text(self, index):
        """根据当前页面更新标签文本"""
        labels = [
            "实时风险仪表盘：显示当日敏感操作占比",
            "敏感数据分布分析饼图：各类型敏感数据占比",
            "词云：高频敏感关键词",
            "用户行为时间线：按小时统计的剪贴板操作次数",
            "日历热力图：每日风险事件总数",
        ]
        if 0 <= index < len(labels):
            self.label.setText(labels[index])
        else:
            self.label.setText("未选择图片")
