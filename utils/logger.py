# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午10:52
# @Site : 
# @file : logger.py
# @Software : PyCharm
# @Description : 

# utils/logger.py
import json
import os

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, data):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"日志已更新: {self.log_file}")