import logging
import colorlog
import os
from datetime import datetime

def get_logger(name="Project"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 如果已经有处理器了就不要重复添加了喵
    if not logger.handlers:
        # --- 1. 设置屏幕输出（带颜色）喵 ---
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s | %(blue)s%(message)s",
            log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white'}
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

        # --- 2. 设置本地文件保存（纯文本）喵 ---
        # 创建一个 logs 文件夹喵
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # 用当前时间命名文件，比如 2026-03-06_19-40.log 喵
        log_filename = datetime.now().strftime("logs/%Y-%m-%d_%H-%M.log")
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        
        # 文件里就不需要颜色代码了，但要加上详细时间喵
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] >> %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = get_logger()