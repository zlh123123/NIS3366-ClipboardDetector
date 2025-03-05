# model_optimizer.py
import torch
import json
import argparse
from transformers import DistilBertForSequenceClassification
import os


script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
base_dir = os.path.dirname(script_dir)  # 获取ModelTrainCode目录
dataset_dir = os.path.join(base_dir, "Dataset")  # 获取Dataset目录
model_path = os.path.join(dataset_dir, "privacy_detection_model.pth")
map_path = os.path.join(dataset_dir, "label_map.json")
model_save_dir = os.path.join(base_dir, "pretrained_models")
optimize_model_dir = os.path.join(dataset_dir, "optimized_privacy_model.onnx")


def optimize_model(
    input_model_path=model_path,
    output_onnx_path=optimize_model_dir,
    pruning_amount=0.2,
    label_map_path=map_path,
    max_len=128,
):
    """
    模型优化流水线
    参数：
        input_model_path: 原始PyTorch模型路径
        output_onnx_path: 优化后ONNX模型保存路径
        pruning_amount: 剪枝比例 (0-1)
        label_map_path: 标签映射文件路径
        max_len: 模型最大输入长度
    """
    # 加载标签映射
    with open(label_map_path) as f:
        label_map = json.load(f)
        num_labels = len(label_map)

    # 1. 加载原始模型
    print("🔄 加载原始模型...")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_save_dir, num_labels=num_labels
    )
    model.load_state_dict(
        torch.load(input_model_path, map_location=torch.device("cpu"))
    )
    model.eval()

    # 2. 动态量化
    print("⚡ 执行动态量化...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # 3. 模型剪枝
    print("✂️ 执行全局剪枝 (比例: {:.0%})...".format(pruning_amount))
    parameters_to_prune = [
        (module, "weight")
        for module in quantized_model.modules()
        if isinstance(module, torch.nn.Linear)
    ]

    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=pruning_amount,
    )

    # 永久移除剪枝掩码
    for module, _ in parameters_to_prune:
        torch.nn.utils.prune.remove(module, "weight")

    # 4. 导出ONNX
    print("📤 导出ONNX模型...")
    dummy_input = (
        torch.randint(0, 10000, (1, max_len)),  # input_ids
        torch.ones(1, max_len),  # attention_mask
    )

    torch.onnx.export(
        quantized_model,
        dummy_input,
        output_onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
        verbose=False,
    )

    print(f"✅ 优化完成！模型已保存至: {output_onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型优化工具")
    parser.add_argument("--input", type=str, default=model_path, help="输入模型路径")
    parser.add_argument(
        "--output",
        type=str,
        default=optimize_model_dir,
        help="输出ONNX路径",
    )
    parser.add_argument("--prune", type=float, default=0.2, help="剪枝比例 (0-1)")
    parser.add_argument("--max_len", type=int, default=128, help="模型最大输入长度")

    args = parser.parse_args()

    optimize_model(
        input_model_path=args.input,
        output_onnx_path=args.output,
        pruning_amount=args.prune,
        max_len=args.max_len,
    )
