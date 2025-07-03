#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具管理模块
负责管理、注册和执行Agent可用的工具
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable, Union
from pydantic import BaseModel, Field, ValidationError

# 从新的base模块导入
from .base import Tool, ToolMetadata, FunctionTool
from supports.logger import logger

class ToolConfig(BaseModel):
    """工具管理器配置"""
    
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


class ToolManager:
    """
    工具管理器
    负责管理、注册和执行Agent可用的工具
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        """
        初始化工具管理器
        
        Args:
            config: 工具配置
        """
        self.config = config or ToolConfig()
        self.tools: Dict[str, Tool] = {}

    
    def register_tool(self, tool: Union[Tool, Callable, List[Tool], List[Callable]], 
                     name: Optional[Union[str, List[str]]] = None,
                     description: Optional[Union[str, List[str]]] = None, 
                     override: bool = False) -> Union[str, List[str]]:
        """
        注册工具，支持单个工具或工具列表
        
        Args:
            tool: 工具对象、函数，或它们的列表
            name: 工具名称或名称列表，如果为None则使用工具的默认名称
            description: 工具描述或描述列表，如果为None则使用工具的默认描述
            override: 是否覆盖已存在的同名工具
            
        Returns:
            工具名称或名称列表
            
        Raises:
            ValueError: 工具已存在且override为False时抛出
            TypeError: 工具类型不正确时抛出
        """
        # 处理列表输入
        if isinstance(tool, list):
            names = []
            # 处理名称和描述参数
            name_list = [None] * len(tool) if name is None else (
                name if isinstance(name, list) else [name] * len(tool)
            )
            desc_list = [None] * len(tool) if description is None else (
                description if isinstance(description, list) else [description] * len(tool)
            )
            
            # 验证参数列表长度匹配
            if len(name_list) != len(tool) or len(desc_list) != len(tool):
                raise ValueError("工具列表、名称列表和描述列表的长度必须相同")
            
            # 注册每个工具
            for t, n, d in zip(tool, name_list, desc_list):
                names.append(self.register_tool(t, n, d, override))
            return names
        
        # 处理单个工具的原有逻辑
        if callable(tool) and not isinstance(tool, Tool):
            tool = FunctionTool(tool)
        
        if not isinstance(tool, Tool):
            raise TypeError(f"工具必须是Tool类型或可调用对象，而不是 {type(tool)}")
        # 更新元数据(如果提供)
        if name:
            tool.metadata.name = name
        
        if description:
            tool.metadata.description = description
        
        # 检查工具是否已存在
        tool_name = tool.metadata.name
        if tool_name in self.tools and not override:
            raise ValueError(f"工具 {tool_name} 已存在，设置override=True来覆盖")
        
        # 注册工具
        self.tools[tool_name] = tool
        
        return tool_name
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        获取工具
        
        Args:
            name: 工具名称
            
        Returns:
            工具对象，如果不存在则返回None
        """
        return self.tools.get(name)
    
    def has_tool(self, name: str) -> bool:
        """
        检查工具是否存在
        
        Args:
            name: 工具名称
            
        Returns:
            是否存在
        """
        return name in self.tools
    
    def remove_tool(self, name: str) -> bool:
        """
        移除工具
        
        Args:
            name: 工具名称
            
        Returns:
            是否成功移除
        """
        if name in self.tools:
            del self.tools[name]
            logger.info(f"已移除工具: {name}")
            return True
        return False
    
    def list_tools(self) -> List[str]:
        """
        列出所有可用工具名称
        
        Returns:
            工具名称列表
        """
        return list(self.tools.keys())
    
    def list_tools_by_category(self) -> Dict[str, List[str]]:
        """
        按类别列出工具
        
        Returns:
            类别到工具名称列表的映射
        """
        categories = {}
        for name, tool in self.tools.items():
            category = tool.metadata.category
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        
        return categories
    
    def list_tools_description(self) -> str:
        """
        获取所有工具的描述
        
        Returns:
            工具描述字符串
        """
        descriptions = []
        
        for name, tool in self.tools.items():
            descriptions.append(
            {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.metadata.description,
                "parameters": tool.metadata.input_schema,
            },
        })
        return descriptions
        # descriptions = []
        # for name, tool in self.tools.items():
        #     desc = f"{name}: {tool.metadata.description}"
            
        #     # 添加输入说明
        #     if tool.metadata.input_schema:
        #         required = tool.metadata.input_schema.get("required", [])
        #         properties = tool.metadata.input_schema.get("properties", {})
                
        #         params = []
        #         for param_name, param_info in properties.items():
        #             param_type = param_info.get("type", "any")
        #             param_desc = param_info.get("description", "")
        #             is_required = param_name in required
                    
        #             param_str = f"{param_name} ({param_type})"
        #             if is_required:
        #                 param_str += " [必需]"
        #             if param_desc:
        #                 param_str += f": {param_desc}"
                    
        #             params.append(param_str)
                
        #         if params:
        #             desc += "\n  参数:\n    " + "\n    ".join(params)
            
        #     descriptions.append(desc)
        
        # return "\n\n".join(descriptions)
    
    def get_tool_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取工具元数据
        
        Args:
            name: 工具名称
            
        Returns:
            工具元数据字典，如果不存在则返回None
        """
        tool = self.get_tool(name)
        if tool:
            return tool.metadata.dict()
        return None
    
    async def execute(self, tool_name: str, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            执行结果
            
        Raises:
            ValueError: 工具不存在时抛出
            ValidationError: 参数验证失败时抛出
            Exception: 工具执行失败时抛出
        """
        # 检查工具是否存在
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"工具 {tool_name} 不存在")
        logger.info(f"调用工具{tool_name}, 输入{kwargs}")

        # 记录开始时间
        start_time = time.time()
        
        # 实现重试逻辑
        retries = 0
        last_error = None
        
        while retries <= self.config.max_retries:
            try:
                if retries > 0:
                    logger.info(f"重试执行工具 {tool_name}，第 {retries} 次")
                    # 等待一段时间再重试
                    time.sleep(self.config.retry_delay)
                
                # 执行工具
                result = await tool.execute(**kwargs)
                
                # 记录执行时间
                elapsed = time.time() - start_time
                logger.info(f"工具 {tool_name} 执行成功，耗时 {elapsed:.2f} 秒")
                return result
                
            except Exception as e:
                last_error = e
                retries += 1
                
                # 如果是验证错误，直接抛出而不重试
                if isinstance(e, ValidationError):
                    logger.error(f"工具 {tool_name} 参数验证失败: {e}")
                    raise
                
                logger.warning(f"工具 {tool_name} 执行失败: {e}")
                
                # 检查是否达到最大重试次数
                if retries > self.config.max_retries:
                    break
        
        # 重试失败
        elapsed = time.time() - start_time
        logger.error(f"工具 {tool_name} 执行失败，已重试 {retries} 次，总耗时 {elapsed:.2f} 秒")
        
        # 重新抛出最后一个错误
        if last_error:
            raise last_error
        else:
            raise Exception(f"工具 {tool_name} 执行失败，未知错误")
    
    def execute_sync(self, tool_name: str, **kwargs) -> Any:
        """
        同步执行工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            执行结果
        """
        import asyncio
        
        # 获取或创建事件循环
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 执行异步函数
        return loop.run_until_complete(self.execute(tool_name, **kwargs)) 