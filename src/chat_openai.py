import openai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .mcp_client import Tool

@dataclass
class Message:
    role: str
    content: str
    tool_calls: Optional[List[Dict]] = None
    
class ChatOpenAI:
    """Chat interface supporting OpenAI, DeepSeek and Qwen models"""

    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        tools: Optional[List[Tool]] = None,
        context: str = ""
    ):
        self.model = model
        self.messages: List[Message] = []
        self.tools = tools or []
        
        # Add system prompt and context if provided
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))
        if context:
            self.messages.append(Message(role="system", content=f"Context:\n{context}"))
            
    async def chat(self, prompt: str = "") -> str:
        """Send chat message and handle tool calls"""
        
        # Add user message
        if prompt:
            self.messages.append(Message(role="user", content=prompt))
            
        # Format messages for API
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
                **({"tool_calls": msg.tool_calls} if msg.tool_calls else {})
            }
            for msg in self.messages
        ]
        
        # Format tools for API 
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            }
            for tool in self.tools
        ] if self.tools else None
        
        # Detect model type and use appropriate API format
        if "deepseek" in self.model:
            # DeepSeek format
            response = await self._call_deepseek_api(messages, tools)
        elif "qwen" in self.model:
            # Qwen format
            response = await self._call_qwen_api(messages, tools)
        else:
            # OpenAI format
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
        return response.choices[0].message.content
        
    async def _call_deepseek_api(self, messages: List[Dict], tools: Optional[List[Dict]]):
        """Call DeepSeek chat API"""
        # DeepSeek specific implementation
        pass
        
    async def _call_qwen_api(self, messages: List[Dict], tools: Optional[List[Dict]]):
        """Call Qwen chat API"""
        # Qwen specific implementation 
        pass

    async def append_tool_result(self, tool_call_id: str, tool_output: str):
        """Append tool execution result to messages"""
        self.messages.append(Message(
            role="tool",
            content=tool_output,
            tool_calls=[{"id": tool_call_id}]
        ))