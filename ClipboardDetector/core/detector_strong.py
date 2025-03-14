import pyperclip
import time
from typing import Optional, Dict, Any, Tuple
import json
from pathlib import Path
import re
from utils import luhn_checksum, validate_chinese_id
from logger import ClipboardLogger
from desktop_notifier import DesktopNotifier, Button, Sound
from desktop_notifier.common import DEFAULT_SOUND, Icon
import asyncio
import ctypes
from ctypes import wintypes
import psutil
import sqlite3


class SensitiveDetector:
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
            rule_path = Path(__file__).parent.parent / "rules/regular/rule_strong.json"
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


async def monitor_clipboard_with_detection(
    detector: SensitiveDetector, interval: float = 0.5
) -> None:
    """
    监控剪贴板内容并进行敏感信息检测
    Args:
        detector: 敏感信息检测器实例
        interval: 检查间隔时间(秒)
    """
    last_content: Optional[str] = None
    logger = ClipboardLogger()
    print("\n开始监控剪贴板，按 Ctrl+C 停止...\n")

    # 加载白名单
    whitelist = load_whitelist()

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

                results = detector.detect(current_content)

                if results:
                    print("\n发现敏感信息:")
                    notification_msg = "发现以下敏感信息:\n"
                    for rule_name, matches in results.items():
                        description = detector.rules[rule_name]["description"]
                        print(f"- {description}: {matches}")
                        notification_msg += f"- {description}\n"

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
    detector = SensitiveDetector()
    asyncio.run(monitor_clipboard_with_detection(detector))
