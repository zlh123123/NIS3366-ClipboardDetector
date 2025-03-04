import re
import math
from typing import Dict, Set, Optional, Union

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


def validate_phone_number(phone: str) -> bool:
    """验证中国手机号码有效性"""
    # 移除可能的空格和连字符
    phone = phone.replace(" ", "").replace("-", "")

    # 中国主要运营商号段前缀映射
    valid_prefixes = {
        # 中国移动
        "134",
        "135",
        "136",
        "137",
        "138",
        "139",
        "150",
        "151",
        "152",
        "157",
        "158",
        "159",
        "182",
        "183",
        "184",
        "187",
        "188",
        "198",
        "147",
        "178",
        "195",
        "172",
        "148",
        "1703",
        "1705",
        "1706",
        # 中国联通
        "130",
        "131",
        "132",
        "155",
        "156",
        "166",
        "185",
        "186",
        "145",
        "175",
        "176",
        "1704",
        "1707",
        "1708",
        "1709",
        # 中国电信
        "133",
        "153",
        "177",
        "180",
        "181",
        "189",
        "199",
        "1700",
        "1701",
        "1702",
        # 虚拟运营商
        "170",
        "171",
    }

    try:
        # 基本格式检查
        if len(phone) != 11 or not phone.isdigit():
            return False

        # 号段验证
        prefix3 = phone[:3]
        prefix4 = phone[:4]

        return prefix3 in valid_prefixes or prefix4 in valid_prefixes
    except Exception:
        return False


def entropy_check(password: str) -> bool:
    """
    检查密码熵值是否足够高
    """
    if not password or len(password) < 8:
        return False

    # 熵值计算
    charset_size = 0
    if any(c.islower() for c in password):
        charset_size += 26
    if any(c.isupper() for c in password):
        charset_size += 26
    if any(c.isdigit() for c in password):
        charset_size += 10
    if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?/" for c in password):
        charset_size += 33

    # 字符集过小
    if charset_size < 30:
        return False

    # 计算熵值: 长度 * log2(字符集大小)
    entropy = len(password) * math.log2(charset_size)

    # 检查是否有重复模式
    for i in range(1, len(password) // 2 + 1):
        for j in range(len(password) - i):
            pattern = password[j : j + i]
            rest = password[j + i :]
            if pattern in rest:
                # 发现重复模式，降低熵值
                entropy *= 0.8

    # 强密码至少需要50比特熵值
    return entropy >= 50


# 常见弱密码集合 (实际使用中应从文件加载更大的数据集)
_COMMON_PASSWORDS: Set[str] = {
    "password",
    "123456",
    "12345678",
    "qwerty",
    "admin",
    "welcome",
    "1234567890",
    "abc123",
    "111111",
    "123123",
    "admin123",
    "letmein",
    "monkey",
    "password1",
    "123456789",
    "dragon",
    "sunshine",
    "princess",
    "qwertyuiop",
    "trustno1",
    "iloveyou",
}


def password_dictionary_check(password: str) -> bool:
    """检查是否为常见弱密码"""
    if not password:
        return False

    # 转为小写进行比较
    lower_pwd = password.lower()

    # 直接匹配
    if lower_pwd in _COMMON_PASSWORDS:
        return True

    # 检查简单变形
    leet_map = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
    normal_pwd = lower_pwd

    # 转换 1337 speak 回普通文本
    for char, num in leet_map.items():
        normal_pwd = normal_pwd.replace(num, char)

    # 检查转换后的密码
    if normal_pwd in _COMMON_PASSWORDS:
        return True

    # 检查常见模式
    if re.match(r"^(?:\d{1,8})$", password):  # 纯数字
        return True
    if re.match(r"^(?:[a-zA-Z]{1,8})$", password):  # 纯字母
        return True

    return False
