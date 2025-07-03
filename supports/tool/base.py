#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具基类模块
定义所有工具的基类和公共功能
"""

import inspect
from typing import Dict, List, Any, Optional, Callable, Union, Type, Set, Tuple
from pydantic import BaseModel, Field, create_model, ValidationError


class ToolMetadata(BaseModel):
    """工具元数据"""
    
    # 基本信息
    name: str
    description: str
    category: str = "general"
    version: str = "1.0.0"
    author: str = ""
    
    # 输入输出规范
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    
    # 使用示例
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    
    # 安全信息
    requires_permissions: List[str] = Field(default_factory=list)
    is_dangerous: bool = False
    warning: Optional[str] = None


class Tool:
    """工具基类，所有工具都应继承此类"""
    
    metadata: ToolMetadata
    
    def __init__(self, metadata: Optional[ToolMetadata] = None):
        """
        初始化工具
        
        Args:
            metadata: 工具元数据
        """
        self.metadata = metadata or self._get_default_metadata()
        self._validate_metadata()
    
    def _get_default_metadata(self) -> ToolMetadata:
        """获取默认元数据"""
        # 从类属性获取元数据，如果未定义则创建默认元数据
        if hasattr(self.__class__, 'metadata'):
            return self.__class__.metadata
        
        return ToolMetadata(
            name=self.__class__.__name__.lower(),
            description=self.__doc__ or f"{self.__class__.__name__} tool"
        )
    
    def _validate_metadata(self) -> None:
        """验证元数据"""
        if not self.metadata.name:
            self.metadata.name = self.__class__.__name__.lower()
        
        if not self.metadata.description:
            self.metadata.description = self.__doc__ or f"{self.__class__.__name__} tool"
    
    async def execute(self, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            执行结果
        """
        raise NotImplementedError("子类必须实现execute方法")
    
    def validate_input(self, **kwargs) -> Dict[str, Any]:
        """
        验证输入参数
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            验证后的参数
            
        Raises:
            ValidationError: 参数验证失败时抛出
        """
        if not self.metadata.input_schema:
            return kwargs
        
        # 创建验证模型
        fields = {}
        
        for field_name, field_schema in self.metadata.input_schema.get("properties", {}).items():
            field_type = self._map_json_type_to_python(field_schema.get("type", "string"))
            field_required = field_name in self.metadata.input_schema.get("required", [])
            field_default = ... if field_required else None
            field_description = field_schema.get("description", "")
            fields[field_name] = (field_type, Field(default=field_default, description=field_description))
        # 创建动态模型
        model_name = f"{self.metadata.name.capitalize()}Input"
        input_model = create_model(model_name, **fields)
        # 验证输入
        try:
            validated = input_model(**kwargs)
            return validated.dict()
        except ValidationError as e:
            # 重新抛出更友好的错误
            errors = {}
            for error in e.errors():
                loc = ".".join(str(l) for l in error["loc"])
                if loc not in errors:
                    errors[loc] = []
                errors[loc].append(error["msg"])
            
            error_msg = f"工具 {self.metadata.name} 输入验证失败: "
            error_details = "; ".join(f"{field}: {', '.join(msgs)}" for field, msgs in errors.items())
            
            raise ValidationError(f"{error_msg}{error_details}", e.errors())
    
    def _map_json_type_to_python(self, json_type: str) -> Any:
        """
        将JSON Schema类型映射到Python类型
        
        Args:
            json_type: JSON类型字符串
            
        Returns:
            对应的Python类型
        """
        type_mapping = {
            "string": Optional[str],
            "integer": Optional[int],
            "number": Optional[float],
            "boolean": Optional[bool],
            "array": Optional[List[Any]],
            "object": Optional[Dict[str, Any]],
            "null": type(None),
        }
        
        return type_mapping.get(json_type, Any)


class FunctionTool(Tool):
    """基于函数的工具，可以将普通函数包装为工具"""
    
    def __init__(self, func: Callable, metadata: Optional[ToolMetadata] = None):
        """
        初始化函数工具
        
        Args:
            func: 要包装的函数
            metadata: 工具元数据
        """
        self.func = func
        
        # 如果未提供元数据，从函数中提取
        if metadata is None:
            metadata = self._extract_metadata_from_func(func)
        
        super().__init__(metadata)
    
    def _extract_metadata_from_func(self, func: Callable) -> ToolMetadata:
        """从函数提取元数据"""
        # 获取函数信息
        name = func.__name__
        doc = func.__doc__ or f"{name} function"
        
        # 提取参数信息
        sig = inspect.signature(func)
        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            # 跳过self参数
            if param_name == "self":
                continue
                
            param_type = Any
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation
            
            # 添加到属性中
            input_schema["properties"][param_name] = {
                "type": self._get_json_type_from_python(param_type),
                "description": f"{param_name} parameter"
            }
            
            # 如果没有默认值，添加到必需列表
            if param.default == inspect.Parameter.empty:
                input_schema["required"].append(param_name)
        
        return ToolMetadata(
            name=name,
            description=doc,
            input_schema=input_schema
        )
    
    def _get_json_type_from_python(self, py_type: Any) -> str:
        """从Python类型获取JSON类型"""
        # 处理常见类型
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null"
        }
        
        # 获取基本类型
        origin = getattr(py_type, "__origin__", py_type)
        
        # 处理泛型类型
        if origin in (list, List):
            return "array"
        elif origin in (dict, Dict):
            return "object"
        elif origin in (tuple, Tuple):
            return "array"
        elif origin in (set, Set):
            return "array"
        elif origin is Union:
            # 如果是可选类型(例如 Optional[str])，使用基本类型
            args = getattr(py_type, "__args__", [])
            if len(args) == 2 and type(None) in args:
                # 找到非None的类型
                other_type = next(arg for arg in args if arg is not type(None))
                return self._get_json_type_from_python(other_type)
        
        return type_mapping.get(origin, "string")
    
    async def execute(self, **kwargs) -> Any:
        """
        执行工具函数
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            执行结果
        """
        # 验证输入
        validated_kwargs = self.validate_input(**kwargs)
        
        # 执行函数
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**validated_kwargs)
        else:
            return self.func(**validated_kwargs) 