import pyperclip
import time
import os
import json
import numpy as np
import re
import asyncio
import ctypes
import psutil
from ctypes import wintypes
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# 深度学习相关导入
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm

# 项目内部模块导入
from utils import luhn_checksum, validate_chinese_id
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
    """基于深度学习的隐私信息检测器"""

    def __init__(self):
        # 设置文件路径
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.dataset_dir = base_dir / "ModelTrainCode" / "Dataset"
        self.model_path = self.dataset_dir / "privacy_detection_model.pth"
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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {self.device}")

            # 加载tokenizer
            pbar.update(30)
            print("加载tokenizer...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_save_dir)

            # 加载模型
            pbar.update(40)
            print("加载模型架构...")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_save_dir, num_labels=len(self.label_map)
            )

            # 加载模型权重
            pbar.update(60)
            print("加载模型权重...")
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()

            # 完成
            pbar.update(100)

        self.max_len = 128  # 需要与训练时保持一致
        print("隐私检测模型加载完成!")

    def predict(self, text, threshold=0.7):
        """使用模型预测文本中的敏感信息"""
        # 文本编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 转换为设备张量
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()[0]

        # 计算概率
        probs = 1 / (1 + np.exp(-logits))  # sigmoid

        # 生成结果
        results = {}
        for idx, prob in enumerate(probs):
            if prob > threshold:
                label_name = self.label_map[idx]
                results[label_name] = float(prob)

        return results


class SensitiveDetector:
    """基于规则的敏感信息检测器"""

    def __init__(self):
        self.notifier = DesktopNotifier()
        self.rules = {}
        self.validators = {
            "luhn_checksum": luhn_checksum,
            "validate_chinese_id": validate_chinese_id,
        }
        self._load_base_rules()

    def _load_base_rules(self):
        """加载基础规则文件"""
        try:
            rule_path = Path(__file__).parent.parent / "rules/regular/rule_base.json"
            with open(rule_path, "r", encoding="utf-8") as f:
                rules = json.load(f)

            # 处理规则
            for name, rule in rules.items():
                self.rules[name] = {
                    "pattern": rule["pattern"],
                    "description": rule["description"],
                    "validator": (
                        self.validators.get(rule["validator"])
                        if rule["validator"]
                        else None
                    ),
                }
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"加载规则文件失败: {e}")
            raise SystemExit(1)

    def add_custom_rules(self, custom_rule_path: str):
        """添加自定义规则"""
        if not custom_rule_path or not Path(custom_rule_path).exists():
            return
        try:
            with open(custom_rule_path, "r", encoding="utf-8") as f:
                custom_rules = json.load(f)
            for name, rule in custom_rules.items():
                self.rules[name] = {
                    "pattern": rule["pattern"],
                    "description": rule.get("description", ""),
                    "validator": None,
                }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"加载自定义规则失败: {e}")

    def detect(self, text: str) -> Dict[str, list]:
        """执行敏感信息检测"""
        if not text:
            return {}

        results = {}
        for rule_name, rule in self.rules.items():
            try:
                regex = re.compile(rule["pattern"])
                matches = regex.findall(text)
                if matches:
                    if rule["validator"]:
                        valid_matches = [m for m in matches if rule["validator"](m)]
                    else:
                        valid_matches = matches
                    if valid_matches:
                        results[rule_name] = valid_matches
            except re.error as e:
                print(f"正则表达式错误 ({rule_name}): {e}")
        return results


class HybridDetector:
    """结合规则和模型的混合检测器"""

    def __init__(self):
        print("初始化混合检测器...")
        self.rule_detector = SensitiveDetector()
        self.model_detector = PrivacyDetector()
        self.notifier = DesktopNotifier()

    def detect(self, text: str) -> Dict[str, Any]:
        """结合规则和模型进行检测"""
        results = {}

        # 使用规则检测器
        rule_results = self.rule_detector.detect(text)
        if rule_results:
            results["rule_based"] = rule_results

        # 使用模型检测器
        model_results = self.model_detector.predict(text)
        if model_results:
            results["model_based"] = model_results

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


async def monitor_clipboard_with_hybrid_detection(
    detector: HybridDetector, interval: float = 0.5
) -> None:
    """
    监控剪贴板内容并进行敏感信息检测
    Args:
        detector: 混合检测器实例
        interval: 检查间隔时间(秒)
    """
    last_content: Optional[str] = None
    logger = ClipboardLogger()
    print("\n开始监控剪贴板，按 Ctrl+C 停止...\n")

    try:
        while True:
            try:
                current_content = pyperclip.paste()
                if not current_content or current_content == last_content:
                    await asyncio.sleep(interval)
                    continue

                # 获取前台进程信息
                process_name, process_path = get_foreground_process_info()

                print("\n" + "=" * 60)
                print(f"检测到新的剪贴板内容:")
                print(f"来源进程: {process_name} ({process_path})")
                print("-" * 60)
                print(current_content)
                print("-" * 60)

                results = detector.detect(current_content)

                if results:
                    print("\n发现敏感信息:")
                    notification_msg = "发现以下敏感信息:\n"

                    # 规则检测结果
                    if "rule_based" in results:
                        print("规则检测:")
                        for rule_name, matches in results["rule_based"].items():
                            description = detector.rule_detector.rules[rule_name][
                                "description"
                            ]
                            print(f"- {description}: {matches}")
                            notification_msg += f"- {description}\n"

                    # 模型检测结果
                    if "model_based" in results:
                        print("模型检测:")
                        for label_name, confidence in results["model_based"].items():
                            print(f"- {label_name}: 置信度 {confidence:.4f}")
                            notification_msg += (
                                f"- {label_name} (置信度 {confidence:.2f})\n"
                            )

                    try:
                        await show_security_notification(notification_msg)
                    except Exception as e:
                        print(f"通知显示失败: {e}")

                    logger.log_clipboard(
                        current_content, results, process_name, process_path
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
    detector = HybridDetector()
    asyncio.run(monitor_clipboard_with_hybrid_detection(detector))
