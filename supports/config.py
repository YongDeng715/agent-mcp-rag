#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置模块
负责加载和管理应用程序配置
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()

class Config:
    """
    配置类
    加载和管理应用程序配置
    """
    
    def __init__(self):
        """初始化配置"""
        # 加载环境变量
        load_dotenv()
        
        # LLM配置
        self.llm_config = {
            # OpenAI配置
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "openai_api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "openai_org_id": os.getenv("OPENAI_ORG_ID"),
            
            # Azure OpenAI配置
            "azure_openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_openai_api_base": os.getenv("AZURE_OPENAI_API_BASE"),
            
            # Anthropic配置
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "anthropic_api_base": os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com"),
            
            # Google配置
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            "google_api_base": os.getenv("GOOGLE_API_BASE", "https://generativelanguage.googleapis.com"),
            
            # HuggingFace配置
            "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
            "huggingface_api_base": os.getenv("HUGGINGFACE_API_BASE", "https://api-inference.huggingface.co/models"),
            
            # 默认模型配置
            "default_model_type": os.getenv("DEFAULT_MODEL_TYPE", "openai"),
            "default_model_name": os.getenv("DEFAULT_MODEL_NAME", "qwen2.5-instruct"),
            
            # 请求配置
            "max_tokens": int(os.getenv("MAX_TOKENS", "8192")),
            "temperature": float(os.getenv("TEMPERATURE", "0.7")),
            "top_p": float(os.getenv("TOP_P", "1.0")),
            "frequency_penalty": float(os.getenv("FREQUENCY_PENALTY", "0.0")),
            "presence_penalty": float(os.getenv("PRESENCE_PENALTY", "0.0")),
            "timeout": int(os.getenv("TIMEOUT", "60")),
            
            # 重试配置
            "max_retries": int(os.getenv("MAX_RETRIES", "3")),
            "retry_delay": int(os.getenv("RETRY_DELAY", "2")),
            
            # 流式输出配置
            "stream": os.getenv("STREAM", "False").lower() in ["true", "1", "yes"],
            
            # 上下文窗口配置
            "max_context_tokens": int(os.getenv("MAX_CONTEXT_TOKENS", "20000")),
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.llm_config
    
    def get_value(self, key: str, default: Optional[Any] = None) -> Any:
        """获取配置值"""
        return os.getenv(key, default)

# 创建全局配置实例
config = Config()


# 以下是Agent配置相关类

class LLMConfig(BaseModel):
    """LLM配置"""
    
    # 配置字典（替代旧版的Config类）
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 模型配置
    model_name: str = Field(default="gpt-3.5-turbo", description="模型名称")
    model_version: str = Field(default="", description="模型版本，如果为空则使用最新版本")
    provider: str = Field(default="openai", description="模型提供商")
    model_type: str = Field(default="openai", description="模型提供商")
    
    # API配置
    api_key: Optional[str] = Field(default=None, description="API密钥")
    api_base: str = Field(default="https://api.openai.com/v1", description="API基础URL")
    organization_id: Optional[str] = Field(default=None, description="组织ID")
    
    # 请求配置
    max_tokens: int = Field(default=1000, description="最大生成Token数")
    temperature: float = Field(default=0.7, description="温度参数，控制随机性")
    top_p: float = Field(default=1.0, description="Top-p参数，控制采样范围")
    presence_penalty: float = Field(default=0.0, description="存在惩罚，避免重复")
    frequency_penalty: float = Field(default=0.0, description="频率惩罚，避免重复")
    
    # 高级配置
    timeout: float = Field(default=30.0, description="请求超时时间(秒)")
    retry_count: int = Field(default=3, description="失败重试次数")
    stop_sequences: List[str] = Field(default_factory=list, description="停止序列")
    
    # 系统消息
    system_message: str = Field(default="", description="系统消息，用于设置助手的角色和行为")
    
    @field_validator('api_key', mode='before')
    @classmethod
    def validate_api_key(cls, v, info):
        """验证API密钥，如果未提供则尝试从环境变量读取"""
        if not v:
            # 根据提供商从环境变量获取API密钥
            provider = info.data.get('provider', 'openai')
            env_var = f"{provider.upper()}_API_KEY"
            v = os.environ.get(env_var, '')
            
            # 尝试从全局配置获取
            if not v and provider == 'openai':
                v = config.get_value('OPENAI_API_KEY', '')
        return v
    
    @field_validator('temperature', 'top_p')
    @classmethod
    def validate_float_range(cls, v, info):
        """验证浮点数范围"""
        field_name = info.field_name
        if field_name == 'temperature' and (v < 0 or v > 2):
            raise ValueError("temperature必须在0到2之间")
        if field_name == 'top_p' and (v <= 0 or v > 1):
            raise ValueError("top_p必须在0到1之间")
        return v
    
    def model_dump(self, **kwargs):
        """转换为字典，与全局配置兼容（替代旧版的dict方法）"""
        result = super().model_dump(**kwargs)
        # 添加与全局配置兼容的键
        if self.provider == 'openai':
            result['openai_api_key'] = self.api_key
            result['openai_api_base'] = self.api_base
            result['openai_org_id'] = self.organization_id
        return result


class MemoryConfig(BaseModel):
    """内存配置"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 存储配置
    storage_dir: str = Field(default="data/memory", description="内存存储目录")
    session_file_format: str = Field(default="{agent_id}.json", description="会话文件格式")
    
    # 内存配置
    max_memory_items: int = Field(default=100, description="每个会话最大内存条目数")
    enable_compression: bool = Field(default=False, description="是否启用压缩存储")
    
    # 回滚配置
    max_history_states: int = Field(default=10, description="保存的最大历史状态数")
    auto_save_interval: int = Field(default=5, description="自动保存间隔(步骤数)")
    
    @field_validator('storage_dir')
    @classmethod
    def create_storage_dir(cls, v):
        """验证并创建存储目录"""
        os.makedirs(v, exist_ok=True)
        return v


class ToolConfig(BaseModel):
    """工具配置"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 工具执行配置
    timeout: float = Field(default=30.0, description="工具执行超时时间(秒)")
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: float = Field(default=1.0, description="重试等待时间(秒)")
    
    # 安全配置
    allow_remote_execution: bool = Field(default=False, description="是否允许远程执行")
    allow_file_access: bool = Field(default=False, description="是否允许文件访问")
    allowed_domains: List[str] = Field(default_factory=list, description="允许访问的域名")
    
    # 资源限制
    max_memory_mb: int = Field(default=100, description="最大内存使用(MB)")
    max_execution_time: float = Field(default=60.0, description="最大执行时间(秒)")
    
    # 已启用的工具
    enabled_tools: List[str] = Field(default_factory=list, description="启用的工具列表，空列表表示启用所有可用工具")


class AgentConfig(BaseModel):
    """Agent配置"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 基本信息
    name: str = Field(default="智能助手", description="Agent名称")
    description: str = Field(default="一个通用智能助手", description="Agent描述")
    version: str = Field(default="1.0.0", description="Agent版本")
    
    # 运行配置
    max_steps: int = Field(default=10, description="最大步骤数")
    timeout: float = Field(default=300.0, description="执行超时时间(秒)，0表示不限制")
    stuck_timeout: float = Field(default=60.0, description="卡住超时时间(秒)，0表示不检测")
    max_stuck_count: int = Field(default=3, description="最大卡住次数")
    
    # 日志配置
    log_level: int = Field(default=logging.INFO, description="日志级别")
    log_file: Optional[str] = Field(default=None, description="日志文件路径，为None则只输出到控制台")
    
    # 组件配置
    llm_config: LLMConfig = Field(default_factory=LLMConfig, description="LLM配置")
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig, description="内存配置")
    tool_config: ToolConfig = Field(default_factory=ToolConfig, description="工具配置")
    
    # 输入输出配置
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="输入数据验证模式")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, description="输出数据验证模式")
    
    @field_validator('log_file')
    @classmethod
    def setup_log_file(cls, v):
        """设置日志文件，如果提供了路径则确保目录存在"""
        if v:
            log_dir = os.path.dirname(v)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        return v
    
    @model_validator(mode='after')
    def check_timeouts(self):
        """检查超时设置的合理性"""
        timeout = self.timeout
        stuck_timeout = self.stuck_timeout
        
        # 如果超时时间大于0，确保卡住超时时间小于总超时时间的一半
        if timeout > 0 and stuck_timeout > 0 and stuck_timeout > timeout / 2:
            self.stuck_timeout = timeout / 2
        
        return self
    
    @classmethod
    def from_json(cls, json_file: str) -> 'AgentConfig':
        """从JSON文件加载配置"""
        with open(json_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return cls.model_validate(config_data)
    
    def to_json(self, json_file: str) -> None:
        """保存配置到JSON文件"""
        config_data = self.model_dump()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    def update(self, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """使用字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    # 对于嵌套配置，递归更新
                    current_value = getattr(self, key)
                    if hasattr(current_value, 'update') and callable(current_value.update):
                        current_value.update(value)
                    else:
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
        return self
    
    def from_global_config(self) -> 'AgentConfig':
        """从全局配置加载值"""
        # 从全局配置更新LLM配置
        global_llm_config = config.get_llm_config()
        
        llm_config_data = {
            "model_name": global_llm_config.get("default_model_name", self.llm_config.model_name),
            "api_key": global_llm_config.get("openai_api_key", self.llm_config.api_key),
            "api_base": global_llm_config.get("openai_api_base", self.llm_config.api_base),
            "organization_id": global_llm_config.get("openai_org_id", self.llm_config.organization_id),
            "max_tokens": global_llm_config.get("max_tokens", self.llm_config.max_tokens),
            "temperature": global_llm_config.get("temperature", self.llm_config.temperature),
            "top_p": global_llm_config.get("top_p", self.llm_config.top_p),
            "presence_penalty": global_llm_config.get("presence_penalty", self.llm_config.presence_penalty),
            "frequency_penalty": global_llm_config.get("frequency_penalty", self.llm_config.frequency_penalty),
            "timeout": global_llm_config.get("timeout", self.llm_config.timeout),
            "retry_count": global_llm_config.get("max_retries", self.llm_config.retry_count),
        }
        
        # 更新配置
        self.llm_config = LLMConfig(**llm_config_data)
        return self


# 创建默认Agent配置实例
default_agent_config = AgentConfig()


def load_agent_config(config_path: Optional[str] = None) -> AgentConfig:
    """
    加载Agent配置
    
    Args:
        config_path: 配置文件路径，如果为None则返回默认配置
        
    Returns:
        Agent配置
    """
    agent_config = AgentConfig.model_validate(default_agent_config.model_dump())
    
    # 从全局配置加载一些值
    agent_config.from_global_config()
    
    # 如果提供了配置文件，从文件加载
    if config_path and os.path.exists(config_path):
        file_config = AgentConfig.from_json(config_path)
        # 合并配置
        for field in file_config.model_fields:
            if hasattr(file_config, field):
                setattr(agent_config, field, getattr(file_config, field))
    
    return agent_config 
