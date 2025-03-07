import pyperclip
import time
import os
import json
import numpy as np
import asyncio
import ctypes
import psutil
import onnxruntime as ort
from ctypes import wintypes
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import sqlite3

# 深度学习相关导入
from transformers import DistilBertTokenizer
from tqdm import tqdm

# 项目内部模块导入
from logger import ClipboardLogger
from desktop_notifier import DesktopNotifier, Button, Sound
from desktop_notifier.common import DEFAULT_SOUND, Icon


class ModelLoadingProgressBar:
    """模型加载进度条显示类"""

    def __init__(self, desc="加载模型中"):
        self.desc = desc

    def __enter__(self):
        self.pbar = tqdm(total=100, desc=self.desc, ncols=100)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.close()

    def update(self, progress):
        self.pbar.update(progress - self.pbar.n)


class PrivacyDetector:
    """基于ONNX的隐私信息检测器"""

    def __init__(self):
        # 设置文件路径
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.dataset_dir = base_dir / "ModelTrainCode" / "Dataset"
        self.model_path = (
            base_dir / "ModelTrainCode" / "onnx_model" / "model.onnx"
        )  # ONNX模型文件
        self.map_path = self.dataset_dir / "label_map.json"
        self.model_save_dir = base_dir / "ModelTrainCode" / "pretrained_models"

        # 加载模型和配置
        with ModelLoadingProgressBar("正在加载隐私检测模型...") as pbar:
            # 加载标签映射
            pbar.update(10)
            with open(self.map_path) as f:
                label_map_inverted = json.load(f)
                self.label_map = {int(v): k for k, v in label_map_inverted.items()}

            # 初始化设备
            pbar.update(20)
            print("检查可用执行提供程序...")
            providers = ort.get_available_providers()
            print(f"可用ONNX执行提供程序: {providers}")

            # 优先使用GPU加速
            if "CUDAExecutionProvider" in providers:
                self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                print("使用CUDA执行提供程序")
            else:
                self.providers = ["CPUExecutionProvider"]
                print("使用CPU执行提供程序")

            # 加载tokenizer
            pbar.update(30)
            print("加载tokenizer...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_save_dir)

            # 加载ONNX模型
            pbar.update(60)
            print(f"加载ONNX模型: {self.model_path}")
            try:
                self.session = ort.InferenceSession(
                    str(self.model_path), providers=self.providers
                )
                # 获取模型的输入输出名称
                self.input_names = [input.name for input in self.session.get_inputs()]
                self.output_names = [
                    output.name for output in self.session.get_outputs()
                ]
                print(f"模型输入: {self.input_names}")
                print(f"模型输出: {self.output_names}")
            except Exception as e:
                print(f"模型加载失败: {e}")
                raise

            # 完成
            pbar.update(100)

        self.max_len = 128  # 需要与训练时保持一致
        print("隐私检测模型加载完成!")

    def detect(self, text, threshold=0.7):
        """使用ONNX模型预测文本中的敏感信息"""
        # 文本编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",  # 使用numpy而非pytorch
        )

        # 准备输入
        onnx_inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        # 检查输入名称是否与模型期望的一致，否则调整
        if set(self.input_names) != set(onnx_inputs.keys()):
            print(f"调整输入名称以匹配模型: {self.input_names}")
            onnx_inputs = {
                name: (
                    encoding["input_ids"]
                    if "input_id" in name
                    else encoding["attention_mask"]
                )
                for name in self.input_names
            }

        # 推理
        outputs = self.session.run(self.output_names, onnx_inputs)

        # 处理输出 - 假设输出是logits
        logits = outputs[0][0]  # 第一个输出的第一个样本

        # 计算概率
        probs = 1 / (1 + np.exp(-logits))  # sigmoid

        # 生成结果
        results = {}
        for idx, prob in enumerate(probs):
            if prob > threshold:
                label_name = self.label_map[idx]
                results[label_name] = float(prob)

        return results


def get_foreground_process_info() -> Tuple[str, str]:
    """
    获取当前前台窗口所属的进程信息

    Returns:
        Tuple[str, str]: (进程名称, 进程路径)
    """
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()

    # 获取进程ID
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

    try:
        process = psutil.Process(pid.value)
        process_name = process.name()
        process_path = process.exe()
        return process_name, process_path
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return "未知进程", ""


async def show_security_notification(message: str) -> None:
    """
    显示桌面安全通知
    Args:
        message: 通知消息内容
    """
    notifier = DesktopNotifier()

    def clear_clipboard():
        pyperclip.copy("")
        print("剪贴板已清空，当前内容:", pyperclip.paste())

    buttons = [
        Button(title="清空剪贴板", on_pressed=clear_clipboard, identifier="clear"),
        Button(
            title="忽略",
            on_pressed=lambda: print("用户已忽略警告"),
            identifier="ignore",
        ),
    ]

    await notifier.send(
        title="剪贴板安全警告",
        message=message,
        buttons=buttons,
        timeout=10,
        sound=DEFAULT_SOUND,
        icon=None,
    )


def load_whitelist(db_path="whitelist.db") -> set:
    """
    加载白名单数据库

    Args:
        db_path: 白名单数据库路径

    Returns:
        set: 白名单进程集合
    """
    whitelist = set()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT app_name FROM whitelist")
        rows = cursor.fetchall()
        whitelist = {row[0] for row in rows}
        conn.close()
        print("白名单加载成功")
    except sqlite3.Error as e:
        print(f"加载白名单失败: {e}")
    return whitelist


async def monitor_clipboard_with_model_detection(
    detector: PrivacyDetector, interval: float = 0.5
) -> None:
    """
    监控剪贴板内容并使用模型进行敏感信息检测
    Args:
        detector: 模型检测器实例
        interval: 检查间隔时间(秒)
    """
    last_content: Optional[str] = None
    logger = ClipboardLogger()
    print("\n开始使用ONNX模型监控剪贴板，按 Ctrl+C 停止...\n")

    # 加载白名单
    whitelist = load_whitelist()
    # print(f"白名单进程: {whitelist}")

    try:
        while True:
            try:
                current_content = pyperclip.paste()
                if not current_content or current_content == last_content:
                    await asyncio.sleep(interval)
                    continue

                # 获取前台进程信息
                process_name, process_path = get_foreground_process_info()

                # 检查进程是否在白名单中
                if process_name in whitelist:
                    print(f"来源进程 {process_name} 在白名单中，跳过检测")
                    # 每次有剪贴板操作都记录到数据库
                    logger.log_clipboard(
                        current_content, {}, process_name, process_path
                    )
                    print("\n已记录到数据库")
                    last_content = current_content
                    await asyncio.sleep(interval)
                    continue

                print("\n" + "=" * 60)
                print(f"检测到新的剪贴板内容:")
                print(f"来源进程: {process_name} ({process_path})")
                print("-" * 60)
                print(current_content)
                print("-" * 60)

                # 使用模型检测
                results = detector.detect(current_content)

                if results:
                    print("\n发现敏感信息:")
                    notification_msg = "发现以下敏感信息:\n"

                    # 模型检测结果
                    for label_name, confidence in results.items():
                        print(f"- {label_name}: 置信度 {confidence:.4f}")
                        notification_msg += (
                            f"- {label_name} (置信度 {confidence:.2f})\n"
                        )

                    try:
                        await show_security_notification(notification_msg)
                    except Exception as e:
                        print(f"通知显示失败: {e}")

                    # 构造记录格式
                    log_result = {"model_based": results}
                    logger.log_clipboard(
                        current_content, log_result, process_name, process_path
                    )
                    print("\n已记录到数据库")
                else:
                    print("\n未发现敏感信息")

                print("=" * 60 + "\n")
                last_content = current_content

            except pyperclip.PyperclipException as e:
                print(f"无法访问剪贴板: {e}")
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"发生未知错误: {e}")
                await asyncio.sleep(interval)

    except KeyboardInterrupt:
        print("\n已停止监控剪贴板")


if __name__ == "__main__":
    detector = PrivacyDetector()
    asyncio.run(monitor_clipboard_with_model_detection(detector))
