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
            # 检查是否需要更新表结构（添加process_name和process_path列）
            cursor = conn.cursor()
            columns_info = cursor.execute(
                "PRAGMA table_info(clipboard_logs)"
            ).fetchall()
            column_names = [column[1] for column in columns_info]

            # 如果表不存在，创建新表
            if not columns_info:
                conn.execute(
                    """CREATE TABLE clipboard_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    content_hash TEXT NOT NULL,
                    risk_level TEXT CHECK(risk_level IN ('low', 'medium', 'high')),
                    preview TEXT,
                    encrypted_data BLOB NOT NULL,
                    process_name TEXT,
                    process_path TEXT
                )"""
                )
            # 如果表存在但没有process相关列，添加这些列
            else:
                if "process_name" not in column_names:
                    conn.execute(
                        "ALTER TABLE clipboard_logs ADD COLUMN process_name TEXT"
                    )
                if "process_path" not in column_names:
                    conn.execute(
                        "ALTER TABLE clipboard_logs ADD COLUMN process_path TEXT"
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

    def log_clipboard(
        self, content, detection_results, process_name="未知", process_path=""
    ):
        """
        记录剪贴板内容

        Args:
            content: 剪贴板内容
            detection_results: 检测结果
            process_name: 来源进程名称
            process_path: 来源进程路径
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        risk_level = self._calculate_risk_level(detection_results)
        preview = self._create_preview(content, detection_results)
        encrypted_data = self.fernet.encrypt(content.encode())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO clipboard_logs 
                (content_hash, risk_level, preview, encrypted_data, process_name, process_path)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    content_hash,
                    risk_level,
                    preview,
                    encrypted_data,
                    process_name,
                    process_path,
                ),
            )
