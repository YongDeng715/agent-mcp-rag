#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证器模块
提供输入和输出数据的验证功能
"""

import json
import logging

from typing import Any, Dict, Optional, Union, List, Type
from pydantic import BaseModel, ValidationError, create_model

# 配置日志记录器
logger = logging.getLogger(__name__)


class ValidationResult:
    """验证结果类"""
    
    def __init__(self, is_valid: bool, data: Any = None, errors: Optional[Dict[str, List[str]]] = None):
        """
        初始化验证结果
        
        Args:
            is_valid: 是否有效
            data: 验证后的数据
            errors: 验证错误信息
        """
        self.is_valid = is_valid
        self.data = data
        self.errors = errors or {}
    
    def __bool__(self) -> bool:
        """布尔值转换，用于条件表达式"""
        return self.is_valid


class InvalidInputError(Exception):
    """无效输入异常"""
    
    def __init__(self, message: str, errors: Optional[Dict[str, List[str]]] = None):
        """
        初始化异常
        
        Args:
            message: 错误消息
            errors: 详细错误信息
        """
        super().__init__(message)
        self.errors = errors or {}


class SchemaValidator:
    """基于Schema的验证器"""
    
    @staticmethod
    def validate_against_schema(data: Any, schema: Dict[str, Any]) -> ValidationResult:
        """
        根据JSON Schema验证数据
        
        Args:
            data: 要验证的数据
            schema: JSON Schema
            
        Returns:
            验证结果
        """
        try:
            # 根据schema创建Pydantic模型
            model = SchemaValidator._create_model_from_schema(schema)
            
            # 验证数据
            validated_data = model.parse_obj(data)
            
            return ValidationResult(is_valid=True, data=validated_data.dict())
            
        except ValidationError as e:
            errors = {}
            for error in e.errors():
                loc = ".".join(str(l) for l in error["loc"])
                if loc not in errors:
                    errors[loc] = []
                errors[loc].append(error["msg"])
            
            logger.error(f"数据验证失败: {errors}")
            return ValidationResult(is_valid=False, errors=errors)
            
        except Exception as e:
            logger.error(f"验证过程中发生异常: {e}")
            return ValidationResult(is_valid=False, errors={"_general": [str(e)]})
    
    @staticmethod
    def _create_model_from_schema(schema: Dict[str, Any]) -> Type[BaseModel]:
        """
        从JSON Schema创建Pydantic模型
        
        Args:
            schema: JSON Schema
            
        Returns:
            Pydantic模型类
        """
        if not isinstance(schema, dict):
            raise ValueError("Schema必须是一个字典")
        
        # 处理基本结构
        title = schema.get("title", "DynamicModel")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # 构建字段定义
        fields = {}
        for prop_name, prop_schema in properties.items():
            field_type = SchemaValidator._map_json_type_to_python(prop_schema.get("type", "string"))
            field_default = ...  # 默认为必填
            
            # 如果不是必填，设置默认值为None
            if prop_name not in required:
                field_default = None
            
            fields[prop_name] = (field_type, field_default)
        
        # 创建模型类
        return create_model(title, **fields)
    
    @staticmethod
    def _map_json_type_to_python(json_type: str) -> Any:
        """
        将JSON Schema类型映射到Python类型
        
        Args:
            json_type: JSON类型字符串
            
        Returns:
            对应的Python类型
        """
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List,
            "object": Dict,
            "null": type(None),
        }
        
        return type_mapping.get(json_type, Any)


class ModelValidator:
    """基于Pydantic模型的验证器"""
    
    @staticmethod
    def validate_against_model(data: Any, model_class: Type[BaseModel]) -> ValidationResult:
        """
        根据Pydantic模型验证数据
        
        Args:
            data: 要验证的数据
            model_class: Pydantic模型类
            
        Returns:
            验证结果
        """
        try:
            # 验证数据
            model_instance = model_class.parse_obj(data)
            return ValidationResult(is_valid=True, data=model_instance.dict())
            
        except ValidationError as e:
            errors = {}
            for error in e.errors():
                loc = ".".join(str(l) for l in error["loc"])
                if loc not in errors:
                    errors[loc] = []
                errors[loc].append(error["msg"])
            
            logger.error(f"数据验证失败: {errors}")
            return ValidationResult(is_valid=False, errors=errors)
            
        except Exception as e:
            logger.error(f"验证过程中发生异常: {e}")
            return ValidationResult(is_valid=False, errors={"_general": [str(e)]})


def validate_input(input_data: Any, schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None) -> Any:
    """
    验证输入数据
    
    Args:
        input_data: 输入数据
        schema: 验证模式，可以是JSON Schema字典或Pydantic模型类
        
    Returns:
        验证后的数据
        
    Raises:
        InvalidInputError: 输入验证失败时抛出
    """
    # 如果没有提供schema，执行基本类型检查
    if schema is None:
        return input_data
    
    # 根据schema类型选择验证方法
    if isinstance(schema, dict):
        result = SchemaValidator.validate_against_schema(input_data, schema)
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        result = ModelValidator.validate_against_model(input_data, schema)
    else:
        raise ValueError(f"不支持的schema类型: {type(schema)}")
    
    # 处理验证结果
    if result.is_valid:
        return result.data
    else:
        error_msg = "输入验证失败"
        if result.errors.get("_general"):
            error_msg = f"{error_msg}: {result.errors['_general'][0]}"
        
        raise InvalidInputError(error_msg, result.errors)


def validate_output(output_data: Any, schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None) -> Any:
    """
    验证输出数据
    
    Args:
        output_data: 输出数据
        schema: 验证模式，可以是JSON Schema字典或Pydantic模型类
        
    Returns:
        验证后的数据
        
    Note:
        此函数实现暂时留空，将在未来实现
    """
    # 暂时不实现，只返回原始数据
    return output_data 
