# 创建一个新文件：fix_torch_streamlit.py
# 将此文件放在项目根目录中

import os
import sys

# 设置环境变量，禁用Streamlit的文件监视功能
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'

# 尝试修复PyTorch和Streamlit的兼容性问题
def patch_torch_for_streamlit():
    try:
        import torch
        
        # 创建一个安全的包装器，防止Streamlit访问torch.classes的特定属性
        class SafeClassesWrapper:
            def __init__(self, original_module):
                self._original = original_module
                
            def __getattr__(self, name):
                # 特殊处理可能导致问题的属性
                if name == '__path__' or name.startswith('_'):
                    if hasattr(self._original, name):
                        attr = getattr(self._original, name)
                        # 如果是一个对象并且有_path属性，返回一个安全的代理
                        if hasattr(attr, '_path'):
                            return type('SafePath', (), {'_path': []})
                        return attr
                    return None
                return getattr(self._original, name)
        
        # 打补丁到torch.classes
        if hasattr(torch, 'classes'):
            torch.classes = SafeClassesWrapper(torch.classes)
            print("已应用PyTorch与Streamlit兼容性补丁")
    except Exception as e:
        print(f"应用PyTorch补丁时出错: {e}")

# 应用补丁
patch_torch_for_streamlit()