#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Agent基类
提供了Agent的核心功能，包括步骤管理、内存管理、卡住状态处理等
"""

import os
import time
import json
import logging
import uuid
import ast
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from .memory_manager import MemoryManager
from .tool.tool_manager import ToolManager
from .llm import LLMInterface
from .validators import validate_input, validate_output
from .exceptions import AgentStuckError, AgentTimeoutError, InvalidInputError
from .config import AgentConfig
from .logger import logger
from .schema import ToolCall
from .schema import ROLE_TYPE, Memory, Message, format_messages, AgentState

class BaseAgent(ABC):
    """
    LLM Agent基类
    提供Agent的核心功能实现
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_interface: Optional[LLMInterface] = None,
        memory_manager: Optional[MemoryManager] = None,
        tool_manager: Optional[ToolManager] = None,
        available_tools: Optional[ToolManager] = None,
    ):
        """
        初始化Agent
        
        Args:
            config: Agent配置
            llm_interface: LLM接口
            memory_manager: 内存管理器
            tool_manager: 工具管理器
        """
        # 初始化配置
        self.config = config or AgentConfig()
        
        # 初始化状态
        self.state = AgentState(
            name=self.config.name,
            description=self.config.description,
            max_stuck_count=self.config.max_stuck_count
        )
        
        # 初始化组件
        self.llm = LLMInterface()
        self.memory = Memory()
        self.tools = tool_manager or ToolManager(self.config.tool_config)
        self.tool_calls: List[ToolCall] = Field(default_factory=list)
        # 初始化日志
        self.available_tools = available_tools or ToolManager(self.config.tool_config)
        self.current_step = 0
        self.max_steps =3 
        self.next_step_prompt = """
Based on user needs, proactively select the most appropriate tool or combination of tools. For complex tasks, you can break down the problem and use different tools step by step to solve it. After using each tool, clearly explain the execution results and suggest the next steps.

If you want to stop the interaction at any point, use the `terminate` tool/function call.
"""
    
    async def run(self, input_data: Any) -> Any:
        """
        运行Agent
        
        Args:
            input_data: 输入数据
            
        Returns:
            处理结果
        """
        try:
            # 验证输入
            validated_input = self._validate_input(input_data)
            message_map = {
                "user": Message.user_message,
                "system": Message.system_message,
                "assistant": Message.assistant_message,
                "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }
            self._init_run(validated_input)

            self.memory.add_message(Message.user_message(input_data))
        
            # 初始化运行状态
            # 运行Agent直到完成或出错
            while self.state.is_running:
                result = await self._execute_steps()

                # 完成运行
                self._finish_run(result)
            return result
            
        except Exception as e:
            self._handle_error(e)
            raise
    
    def _validate_input(self, input_data: Any) -> Any:
        """验证输入数据"""
        try:
            return validate_input(input_data, self.config.input_schema)
        except Exception as e:
            raise InvalidInputError(f"输入验证失败: {e}")
    
    def is_stuck(self, duplicate_threshold=1) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= duplicate_threshold
    
    def _init_run(self, input_data: Any) -> None:
        """初始化运行"""
        # 重置状态
        if self.state.is_running:
            logger.warning("Agent已经在运行中，重置状态")
            self.state.reset()
        
        # 设置运行状态
        self.state.is_running = True
        self.state.start_time = time.time()
        self.state.last_progress_time = time.time()
        # 避免重复调用memory
        self.memory = Memory()
        # 记录输入
        self.state.input_history.append({
            "timestamp": self.state.start_time,
            "data": input_data
        })
        
        logger.info(f"Agent {self.state.name} 开始运行")

    async def _execute_steps(self) -> Any:
        """执行步骤直到完成"""
        results: List[str] = []
        while (
                self.current_step < self.max_steps 
            ):
            # 检查是否超时
            self.current_step += 1
            logger.info(f"Executing step {self.current_step}/{self.max_steps}")
            if self._check_timeout():
                raise AgentTimeoutError(f"Agent执行超时，已运行 {time.time() - self.state.start_time:.2f} 秒")
            try:
                step_result = await self.step()
                
                # check duplicate response
                logger.info(f"stuck status:{self.is_stuck()}")
                # 更新进度时间
                self.state.last_progress_time = time.time()
                self.state.stuck_count = 0
                results.append(f"Step {self.current_step}: {step_result}")
                    
            except Exception as e:
                if self._handle_step_error(e):
                    continue
                else:
                    raise
            if self.current_step >= self.max_steps:
                self.current_step = 0
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
                break
        
        return "\n".join(results) if results else "No steps executed"
    
    def _finish_run(self, result: Any) -> None:
        """完成运行"""
        self.state.is_running = False
        self.state.end_time = time.time()
        
        # 记录输出
        self.state.output_history.append({
            "timestamp": self.state.end_time,
            "data": result
        })
        
        # 保存内存
        # self.memory.save_session(self.state.agent_id)
        
        duration = self.state.end_time - self.state.start_time
        logger.info(f"Agent {self.state.name} 完成运行，耗时 {duration:.2f} 秒，步骤数 {self.state.current_step}")
    
    def _handle_error(self, error: Exception) -> None:
        """处理错误"""
        logger.error(f"Agent执行错误: {error}")
        
        # 记录错误状态
        if self.state.is_running:
            self.state.is_running = False
            self.state.end_time = time.time()
    
    def _handle_step_error(self, error: Exception) -> bool:
        """
        处理步骤执行错误
        
        Returns:
            是否继续执行
        """
        logger.warning(f"步骤执行错误: {error}")
        
        # 增加卡住计数
        self.state.stuck_count += 1
        
        # 检查是否达到最大卡住次数
        if self.state.stuck_count >= self.state.max_stuck_count:
            logger.error(f"Agent卡住，已达到最大卡住次数 {self.state.max_stuck_count}")
            return False
        
        # 尝试恢复
        try:
            self._recover_from_stuck()
            return True
        except Exception as e:
            logger.error(f"恢复失败: {e}")
            return False
    
    def _check_timeout(self) -> bool:
        """检查是否超时"""
        if self.config.timeout <= 0:
            return False
            
        elapsed = time.time() - self.state.start_time
        return elapsed > self.config.timeout
    
    def _check_stuck(self) -> bool:
        """检查是否卡住"""
        if self.config.stuck_timeout <= 0:
            return False
            
        elapsed = time.time() - self.state.last_progress_time
        return elapsed > self.config.stuck_timeout
    
    def _recover_from_stuck(self) -> None:
        """从卡住状态恢复"""
        logger.info(f"尝试从卡住状态恢复，当前卡住计数: {self.state.stuck_count}")
        
        # 实现恢复策略，例如回退到上一个状态，重试等
        # 这里只是一个简单的示例
        if self.memory.can_rollback(self.state.agent_id):
            logger.info("回退到上一个状态")
            self.memory.rollback(self.state.agent_id)
    
    def _check_completion(self, result: Any) -> bool:
        """
        检查是否完成
        
        Args:
            result: 步骤结果
            
        Returns:
            是否完成
        """
        # 子类可以重写此方法以实现自定义的完成检查
        return self.state.current_step >= self.config.max_steps or result is not None
    
    def save_state(self, file_path: str) -> None:
        """
        保存Agent状态到文件
        
        Args:
            file_path: 文件路径
        """
        state_dict = self.state.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Agent状态已保存到 {file_path}")
    
    def load_state(self, file_path: str) -> None:
        """
        从文件加载Agent状态
        
        Args:
            file_path: 文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)
            
        self.state = AgentState.from_dict(state_dict)
        logger.info(f"Agent状态已从 {file_path} 加载")
    
    async def step(self) -> str:
        """Execute a single step: think and act."""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()
    
    async def think(self) -> Dict[str, Any]:
        """
        思考阶段，生成下一步行动计划
        
        Args:
            context: 上下文信息
            
        Returns:
            思考结果
        """
        should_act = False  
        
        user_msg = Message.user_message(self.next_step_prompt)
        self.messages += [user_msg]
        context =  format_messages(self.messages)

    
        think_prompt = self._build_think_prompt(context)

        self.messages += [Message.user_message(think_prompt)]

        response = self.llm.generate(
            prompt=think_prompt,
            context=context,
            tools=self.tools.list_tools_description(),
            tool_choice="required"
        )
        self.tool_calls = tool_calls = (
            response["tool_calls"] if response and "tool_calls" in response else []
        )

        
        content = response["content"] if response and response["content"] else ""
        formatted_calls = [
            {"id": call["id"], "function": call["function"], "type": "function"}
            for call in tool_calls
        ]
        should_act =  bool(self.tool_calls)
        
        assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=formatted_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
        self.memory.add_message(assistant_msg)
        return should_act
    
    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            self._current_base64_image = None
            result = await self.execute_tool(command)
            
            logger.info(
                f"🎯 Tool '{command['function']['name']}' completed its mission! Result: {result}"
            )   
            results.append(result)
        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command["function"] or not command["function"]["name"]:
            return "Error: Invalid command format"

        name = command["function"]["name"]

        if name not in self.available_tools.list_tools():
            return f"Error: Unknown tool '{name}'"
        logger.info(f"function_name:{name}")
        try:
            # Parse arguments
            args = json.loads(command["function"]["arguments"] or "{}")
            
            ## checkpoint: may lead bugs
            try:
                result = ast.literal_eval(args)
                # 确保结果是字典类型
                if isinstance(result, dict):
                    args = result
            except (SyntaxError, ValueError):
                pass
            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            logger.info(args)
            logger.info(f"name: {name}, tool_input: {args}, type: {type(args)}")
            result = await self.available_tools.execute(name, **args)
            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"
        
    def _build_think_prompt(self, context: Dict[str, Any]) -> str:
        """构建思考提示"""
        return f"""
You are an agent that can execute tool calls
        
context:
{json.dumps(context, ensure_ascii=False, indent=2)}

"""
    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value
        
        



    