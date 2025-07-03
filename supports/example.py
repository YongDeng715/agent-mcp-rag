#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例：使用LLM接口
演示如何使用LLM接口生成文本，配置从.env文件中加载
"""

import logging
from .llm import LLMInterface, LLMConfig, ModelType
from .schema import Message
from .config import config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """主函数"""
    # 从配置中获取LLM配置
    llm_config = config.get_llm_config()
    model_name = llm_config.get("default_model_name")
    # 打印当前配置信息
    print("当前LLM配置:")
    print(f"- 模型类型: {llm_config.get('default_model_type')}")
    print(f"- 模型名称: {model_name}")
    
    # 如果API密钥未设置，提示用户
    if not llm_config.get("openai_api_key") or llm_config.get("openai_api_key") == "your_openai_api_key_here":
        print("警告: 尚未设置OpenAI API密钥，请在.env文件中设置OPENAI_API_KEY")
        return
    
    # 创建LLM配置
    config_obj = LLMConfig(
        model_type=ModelType(llm_config.get("default_model_type")),
        model_name=model_name
    )

    # 创建LLM接口
    llm = LLMInterface(config=config_obj)
    
    # 用户提问
    prompt = "解释一下什么是大语言模型？"

    system_message = Message.system_message(
        content="你是一个专业的AI助手，请用回答用户的问题, 自行选择语言."
    )

    user_message = Message.user_message(
        content=prompt,
    )


    messages = [system_message, user_message]
    messages = [msg.to_dict() for msg in messages]
    
    # 生成回答
    print(f"\n问题: {prompt}")
    print("思考中...")
    
    try:
        # 调用LLM生成回答
        response = llm.generate(
            prompt=None,
            context=messages,
        )
        
        # 输出生成结果
        print("\n回答:")
        print(response.get("content", "无法获取回答"))
        
        # 输出使用情况
        usage = response.get("usage", {})
        print(f"\n使用情况:")
        print(f"- 提示Token数: {usage.get('prompt_tokens', 'N/A')}")
        print(f"- 生成Token数: {usage.get('completion_tokens', 'N/A')}")
        print(f"- 总Token数: {usage.get('total_tokens', 'N/A')}")
    except Exception as e:
        print(f"生成失败: {e}")

if __name__ == "__main__":
    main() 
