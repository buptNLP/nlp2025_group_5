from abc import ABC, abstractmethod
from typing import Any, Dict, List
import time
import streamlit as st

class BaseService(ABC):
    """服务基类"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.is_initialized = False
        self.performance_metrics = {
            'total_calls': 0,
            'total_time': 0,
            'avg_time': 0,
            'errors': 0
        }
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化服务"""
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """处理请求的核心方法"""
        pass
    
    def call(self, *args, **kwargs) -> Dict[str, Any]:
        """统一的服务调用接口，包含性能监控"""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return {
                        'success': False,
                        'error': f'{self.service_name} 初始化失败',
                        'data': None,
                        'metrics': self.get_metrics()
                    }
            
            result = self.process(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 更新性能指标
            self._update_metrics(execution_time, success=True)
            
            return {
                'success': True,
                'error': None,
                'data': result,
                'metrics': {
                    'execution_time': execution_time,
                    'service': self.service_name
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, success=False)
            
            return {
                'success': False,
                'error': str(e),
                'data': None,
                'metrics': {
                    'execution_time': execution_time,
                    'service': self.service_name
                }
            }
    
    def _update_metrics(self, execution_time: float, success: bool):
        """更新性能指标"""
        self.performance_metrics['total_calls'] += 1
        self.performance_metrics['total_time'] += execution_time
        self.performance_metrics['avg_time'] = (
            self.performance_metrics['total_time'] / 
            self.performance_metrics['total_calls']
        )
        if not success:
            self.performance_metrics['errors'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self.performance_metrics,
            'error_rate': (
                self.performance_metrics['errors'] / 
                max(self.performance_metrics['total_calls'], 1)
            ) * 100
        }