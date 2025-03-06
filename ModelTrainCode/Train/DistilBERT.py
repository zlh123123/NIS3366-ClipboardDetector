import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import json
import os
from optimum.onnxruntime import ORTModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
base_dir = os.path.dirname(script_dir)  # 获取ModelTrainCode目录
dataset_dir = os.path.join(base_dir, "Dataset")  # 获取Dataset目录

# 配置参数
MAX_LEN = 128
BATCH_SIZE = 128
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

# 创建TensorBoard日志目录
logs_dir = os.path.join(base_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
writer = SummaryWriter(logs_dir)

# 创建图表保存目录
charts_dir = os.path.join(base_dir, "charts")
os.makedirs(charts_dir, exist_ok=True)

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

# 用于记录训练损失的列表
losses = []
epochs = []

# 训练循环
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    # 使用tqdm创建进度条
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    batch_losses = []
    for i, batch in enumerate(progress_bar):
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

        batch_loss = loss.item()
        total_loss += batch_loss
        batch_losses.append(batch_loss)

        # 更新进度条显示当前批次损失
        progress_bar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

        # 将批次损失添加到TensorBoard
        global_step = epoch * len(train_loader) + i
        writer.add_scalar("Loss/batch", batch_loss, global_step)

    avg_loss = total_loss / len(train_loader)
    epochs.append(epoch + 1)
    losses.append(avg_loss)

    # 将平均损失添加到TensorBoard
    writer.add_scalar("Loss/epoch", avg_loss, epoch)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

    # 每5个epoch保存一次损失图表，或者在训练结束时
    if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
        # 显示中文
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker="o", linestyle="-", color="b")
        plt.title("训练损失曲线")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(charts_dir, f"loss_curve_epoch_{epoch+1}.png"))
        plt.close()

model_path = os.path.join(dataset_dir, "privacy_detection_model.pth")
# 保存模型
torch.save(model.state_dict(), model_path)

# 训练结束后，保存最终的损失曲线
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker="o", linestyle="-", color="b")
plt.title("最终训练损失曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(os.path.join(charts_dir, "final_loss_curve.png"))
plt.close()

# 关闭TensorBoard writer
writer.close()

# 训练后添加量化代码
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8  # 量化所有Linear层
)

# 保存量化模型
quant_model_path = os.path.join(dataset_dir, "quantized_privacy_model.pth")
torch.save(quantized_model.state_dict(), quant_model_path)

# 转换并保存ONNX模型(使用训练好的模型，而非从目录加载)
onnx_model_dir = os.path.join(base_dir, "onnx_model")
os.makedirs(onnx_model_dir, exist_ok=True)

# 保存训练好的模型以用于ONNX导出
model.save_pretrained(model_save_dir)

# 使用Hugging Face Optimum库转换
ORTModelForSequenceClassification.from_pretrained(
    model_save_dir, export=True
).save_pretrained(onnx_model_dir)

# 添加ONNX模型验证步骤
print(f"ONNX模型已保存至: {onnx_model_dir}")
