import json
import re
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

# 假设 detector_base.py 中的 SensitiveDetector 类可用
try:
    from ClipboardDetector.core.detector_base import SensitiveDetector
except ImportError:
    print("无法导入 SensitiveDetector 类。请确保路径正确。")
    SensitiveDetector = None  # 为了避免后续代码出错，这里赋值为 None


def evaluate_model(data_path: str, detector: SensitiveDetector) -> Dict[str, float]:
    """
    评估敏感信息检测模型在给定数据集上的性能。

    Args:
        data_path (str): 数据集文件路径，每行包含一个 JSON 对象，
                         对象包含 "text" 字段（待检测文本）和 "labels" 字段（实际标签列表）。
        detector (SensitiveDetector): 用于检测敏感信息的检测器实例。

    Returns:
        Dict[str, float]: 包含漏报率和错报率的字典。
    """
    if not detector:
        return {"漏报率": 0.0, "错报率": 0.0}

    total_samples = 0
    false_negatives = 0  # 漏报
    false_positives = 0  # 错报
    total_actual_positives = 0

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                text = data.get("text", "")
                actual_labels = data.get("labels", [])

                total_samples += 1
                total_actual_positives += len(actual_labels)

                # 模型预测
                predicted_matches = detector.detect(text)
                predicted_labels = list(predicted_matches.keys())

                # 计算漏报和错报
                for label in actual_labels:
                    if label not in predicted_labels:
                        false_negatives += 1

                for label in predicted_labels:
                    if label not in actual_labels:
                        false_positives += 1

            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
            except Exception as e:
                print(f"处理行时发生错误: {e}")

    # 计算漏报率和错报率
    miss_rate = (
        false_negatives / total_actual_positives if total_actual_positives > 0 else 0.0
    )
    false_alarm_rate = (
        false_positives / total_samples if total_samples > 0 else 0.0
    )  # 错报率

    return {"漏报率": miss_rate, "错报率": false_alarm_rate}


def visualize_results(results: Dict[str, float]):
    """
    创建条形图来可视化漏报率和错报率。

    Args:
        results (Dict[str, float]): 包含漏报率和错报率的字典。
    """
    labels = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=["skyblue", "salmon"])
    plt.ylabel("比率")
    plt.title("模型评估结果")
    plt.ylim(0, 1)  # 假设比率在 0 到 1 之间
    plt.show()


if __name__ == "__main__":
    # 替换为你的数据路径
    data_file_path = "ModelTrainCode/Dataset/valid/data.jsonl"
    data_file_path = Path(__file__).parent.parent / data_file_path
    # 创建检测器实例
    detector = SensitiveDetector()

    # 评估模型
    evaluation_results = evaluate_model(str(data_file_path), detector)

    # 打印结果
    print("评估结果:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")

    # 可视化结果
    visualize_results(evaluation_results)
