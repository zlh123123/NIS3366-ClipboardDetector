import asyncio
from typing import Dict, Optional, Any
import pyperclip

from detector_base import SensitiveDetector, get_foreground_process_info
from detector_model import PrivacyDetector, show_security_notification
from logger import ClipboardLogger


class HybridDetector:
    """结合正则规则和深度学习模型的混合检测器"""

    def __init__(self):
        """初始化规则检测器和模型检测器"""
        self.rule_detector = SensitiveDetector()
        self.model_detector = PrivacyDetector()

    def add_custom_rules(self, custom_rule_path: str):
        """添加自定义规则到规则检测器"""
        self.rule_detector.add_custom_rules(custom_rule_path)

    def detect(self, text: str) -> Dict[str, Any]:
        """
        使用规则和模型进行混合检测

        Args:
            text: 要检测的文本内容

        Returns:
            Dict: 包含规则检测和模型检测结果的字典
        """
        if not text:
            return {}

        # 第一步：使用规则检测
        rule_results = self.rule_detector.detect(text)

        # 第二步：使用模型检测
        model_results = self.model_detector.detect(text)

        # 合并结果
        combined_results = {
            "rule_based": rule_results,
            "model_based": model_results,
        }

        return combined_results


async def monitor_clipboard_with_hybrid_detection(
    detector: HybridDetector, interval: float = 0.5
) -> None:
    """
    监控剪贴板内容并使用混合方式进行敏感信息检测

    Args:
        detector: 混合检测器实例
        interval: 检查间隔时间(秒)
    """
    last_content: Optional[str] = None
    logger = ClipboardLogger()
    print("\n开始使用混合检测方式监控剪贴板，按 Ctrl+C 停止...\n")

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

                # 使用混合检测
                results = detector.detect(current_content)

                rule_results = results.get("rule_based", {})
                model_results = results.get("model_based", {})

                # 记录已经由规则检测到的类型，避免模型检测结果重复
                detected_categories = set()
                has_detection = bool(rule_results or model_results)

                if has_detection:
                    print("\n发现敏感信息:")
                    notification_msg = "发现以下敏感信息:\n"

                    # 处理规则检测结果
                    if rule_results:
                        print("\n基于规则的检测结果:")
                        for rule_name, matches in rule_results.items():
                            description = detector.rule_detector.rules[rule_name][
                                "description"
                            ]
                            print(f"- {description}: {matches}")
                            notification_msg += f"- {description}\n"
                            # 记录已检测的类型
                            detected_categories.add(description.lower())

                    # 处理模型检测结果，排除已由规则检测到的类型
                    if model_results:
                        print("\n基于模型的检测结果:")
                        for label_name, confidence in model_results.items():
                            # 检查是否已经由规则检测到同类信息
                            if label_name.lower() not in detected_categories:
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
