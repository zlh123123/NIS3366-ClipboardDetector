import pyperclip
import time
from typing import Optional, Dict, Any
import json
from pathlib import Path
import re


def luhn_checksum(card_number: str) -> bool:
    """银行卡号Luhn算法校验"""
    digits = list(map(int, card_number.replace(" ", "").replace("-", "")))
    odd_digits = digits[-1::-2]  # 从右开始，奇数位
    even_digits = digits[-2::-2]  # 从右开始，偶数位
    total = sum(odd_digits)
    for d in even_digits:
        total += sum(divmod(d * 2, 10))
    return total % 10 == 0


def validate_chinese_id(id_number: str) -> bool:
    """身份证号码验证"""
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_codes = "10X98765432"

    try:
        # 计算校验码
        sum = 0
        for i in range(17):
            sum += int(id_number[i]) * weights[i]
        check = check_codes[sum % 11]
        return str(check) == id_number[-1].upper()
    except (ValueError, IndexError):
        return False


PRESET_RULES: Dict[str, Dict[str, Any]] = {
    "credit_card": {
        "pattern": r"\b(?:\d[ -]*?){13,16}\b",
        "validator": luhn_checksum,
        "description": "信用卡号",
    },
    "chinese_id": {
        "pattern": r"\b[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b",
        "validator": validate_chinese_id,
        "description": "身份证号",
    },
    "phone_number": {
        "pattern": r"\b(?:1[3-9]\d{9})\b",
        "validator": None,
        "description": "手机号码",
    },
    "password": {
        "pattern": r"(?=.*[!@#$%^&*])(?=.*[a-zA-Z]).{8,}",
        "validator": None,
        "description": "密码",
    },
    "btc_address": {
        "pattern": r"\b(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b",
        "validator": None,
        "description": "比特币地址",
    },
    "api_key": {
        "pattern": r"\b(sk|pk)-[a-zA-Z0-9]{32,}\b",
        "validator": None,
        "description": "API密钥",
    },
    "email": {
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "validator": None,
        "description": "电子邮箱",
    },
}


class SensitiveDetector:
    def __init__(self, custom_rule_path: str = None):
        self.rules = PRESET_RULES.copy()
        self._load_custom_rules(custom_rule_path)

    def _load_custom_rules(self, path: str):
        """加载自定义规则JSON文件"""
        if not path or not Path(path).exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
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


def monitor_clipboard_with_detection(
    detector: SensitiveDetector, interval: float = 0.5
) -> None:
    """
    监控剪贴板内容并进行敏感信息检测
    Args:
        detector: 敏感信息检测器实例
        interval: 检查间隔时间(秒)
    """
    last_content: Optional[str] = None
    print("\n开始监控剪贴板，按 Ctrl+C 停止...\n")

    try:
        while True:
            try:
                current_content = pyperclip.paste()
                if current_content and current_content != last_content:
                    print("\n" + "=" * 60)
                    print(f"检测到新的剪贴板内容:")
                    print("-" * 60)
                    print(current_content)
                    print("-" * 60)

                    results = detector.detect(current_content)
                    if results:
                        print("\n发现敏感信息:")
                        for rule_name, matches in results.items():
                            description = detector.rules[rule_name]["description"]
                            print(f"- {description}: {matches}")
                    else:
                        print("\n未发现敏感信息")

                    print("=" * 60 + "\n")
                    last_content = current_content

            except pyperclip.PyperclipException as e:
                print(f"无法访问剪贴板: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n已停止监控剪贴板")


if __name__ == "__main__":
    detector = SensitiveDetector()
    monitor_clipboard_with_detection(detector)
