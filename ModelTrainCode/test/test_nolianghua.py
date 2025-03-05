import torch
import json
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
base_dir = os.path.dirname(script_dir)  # 获取ModelTrainCode目录
dataset_dir = os.path.join(base_dir, "Dataset")  # 获取Dataset目录
model_path = os.path.join(dataset_dir, "privacy_detection_model.pth")
map_path = os.path.join(dataset_dir, "label_map.json")
model_save_dir = os.path.join(base_dir, "pretrained_models")


class PrivacyDetector:
    def __init__(self, model_path=model_path):
        # 加载标签映射
        with open(map_path) as f:
            label_map_inverted = json.load(f)
            # 反转键值对，以适应当前的映射文件
            self.label_map = {int(v): k for k, v in label_map_inverted.items()}

        # 初始化模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_save_dir, num_labels=len(self.label_map)
        )

        # 加载训练好的权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 初始化tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_save_dir)
        self.max_len = 128  # 需要与训练时保持一致

    def predict(self, text, threshold=0.7):
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


# 使用示例
if __name__ == "__main__":
    detector = PrivacyDetector()

    test_texts = [
        "我的信用卡号是4111 1111 1111 1111，邮箱test@example.com",
        "密码是password123，联系电话13800138000",
        "这是一个没有任何敏感信息的普通文本",
        "import pyperclip from winotify import Notification, audio 320681200404110014 15896233393 15896233393",
        "100200300",
        "ssh -p 48077 root@connect.nmb1.seetacloud.com",
        "剪贴板内容安全监控器",
        "https://www.oryoy.com/news/jie-mi-ubuntu-xi-tong-xia-sh-jiao-ben-shuang-ji-zhi-xing-de-zhong-ji-zhi-nan-gao-bie-ming-ling-xing.html",
    ]

    for text in test_texts:
        print(f"输入文本：{text}")
        print("检测结果：")
        results = detector.predict(text)
        print(results if results else "未检测到敏感信息")
        print("-" * 50)
