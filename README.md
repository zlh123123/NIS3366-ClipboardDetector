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

