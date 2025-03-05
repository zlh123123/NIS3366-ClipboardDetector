import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
base_dir = os.path.dirname(script_dir)  # 获取ModelTrainCode目录
dataset_dir = os.path.join(base_dir, "Dataset")  # 获取Dataset目录

# 配置参数
MAX_LEN = 128
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5

# 标签映射字典
LABEL_MAP = {
    0: "信用卡号",
    1: "身份证号",
    2: "手机号码",
    3: "高强度密码",
    4: "低强度密码",
    5: "比特币地址",
    6: "API密钥",
    7: "电子邮箱",
}


map_path = os.path.join(dataset_dir, "label_map.json")
# 标签映射
with open(map_path, "w") as f:
    json.dump({v: k for k, v in LABEL_MAP.items()}, f)


class PrivacyDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len):
        self.data = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {v: k for k, v in LABEL_MAP.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        labels = [self.label_map[l] for l in item["labels"]]

        # 多标签编码
        multi_hot = torch.zeros(len(LABEL_MAP))
        for l in labels:
            multi_hot[l] = 1.0

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": multi_hot,
        }


# 设置模型保存路径
model_save_dir = os.path.join(base_dir, "pretrained_models")
os.makedirs(model_save_dir, exist_ok=True)  # 确保目录存在

# 初始化模型和tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_save_dir)
model = DistilBertForSequenceClassification.from_pretrained(
    model_save_dir,
    num_labels=len(LABEL_MAP),
    problem_type="multi_label_classification",
)

# 将下载的模型保存到指定路径
tokenizer.save_pretrained(model_save_dir)
model.save_pretrained(model_save_dir)

data_path = os.path.join(dataset_dir, "train/data.jsonl")
# 数据加载器
train_dataset = PrivacyDataset(data_path, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.BCEWithLogitsLoss()

# 训练循环
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

model_path = os.path.join(dataset_dir, "privacy_detection_model.pth")
# 保存模型
torch.save(model.state_dict(), model_path)
