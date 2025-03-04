import sqlite3
import hashlib
from cryptography.fernet import Fernet
import os


class ClipboardLogger:
    def __init__(self, db_path="clipboard.db"):
        self.db_path = db_path
        self.key = self._get_or_create_key()
        self.fernet = Fernet(self.key)
        self._init_db()

    def _get_or_create_key(self):
        """获取或创建加密密钥"""
        key_file = "encryption.key"
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            return key

    def _init_db(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS clipboard_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                content_hash TEXT NOT NULL,
                risk_level TEXT CHECK(risk_level IN ('low', 'medium', 'high')),
                preview TEXT,
                encrypted_data BLOB NOT NULL
            )"""
            )

    def _calculate_risk_level(self, results):
        """根据检测结果计算风险等级"""
        if not results:
            return "low"
        if len(results) >= 2:
            return "high"
        if len(results) >= 1:
            return "medium"
        return "low"

    def _create_preview(self, content, results):
        """创建脱敏预览"""
        preview = content
        for matches in results.values():
            for match in matches:
                preview = preview.replace(match, "*" * len(match))
        return preview[:500] + "..." if len(preview) > 500 else preview

    def log_clipboard(self, content, detection_results):
        """记录剪贴板内容"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        risk_level = self._calculate_risk_level(detection_results)
        preview = self._create_preview(content, detection_results)
        encrypted_data = self.fernet.encrypt(content.encode())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO clipboard_logs (content_hash, risk_level, preview, encrypted_data)
                VALUES (?, ?, ?, ?)""",
                (content_hash, risk_level, preview, encrypted_data),
            )
