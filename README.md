# NIS3366-ClipboardDetector
剪贴板内容安全监控器，实时监控剪贴板内容，自动屏蔽敏感信息（如银行卡号、密码），防止意外泄露。

---

## 部署指南

```sh
# 创建Python 3.12虚拟环境，以VENV为例，也可使用conda等
python -m venv ClipboardDetectorVENV

# 激活虚拟环境后安装依赖
pip install -r requirements.txt				# 仅运行主程序
pip install -r requirements_train.txt		# 除运行主程序外，还需运行模型训练程序

# 运行程序
python NIS3366-ClipboardDetector/ClipboardDetector/ui/demo.py
```

> 若需要使用深度模型，可从[夸克网盘](https://pan.quark.cn/s/4a3298184f1e)或[交大云盘](
> https://pan.sjtu.edu.cn/web/share/b7014edc9de2e9e1b22b7a9128b0e654)中下载模型，并将`privacy_detection_model.pth`存放于`NIS3366-ClipboardDetector\ModelTrainCode\Dataset`下，将`onnx_model`文件夹存放于`NIS3366-ClipboardDetector\ModelTrainCode`下
