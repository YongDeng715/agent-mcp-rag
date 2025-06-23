from typing import List, Optional

from .mcp_client import MCPClient 
from .chat_openai import ChatOpenAI
from .utils import log_title

class Agent:
    """Agent class that integrates MCP and LLM capabilities"""
    
    def __init__(
        self, 
        model: str,
        mcp_clients: List[MCPClient],
        system_prompt: str = "",
        context: str = ""
    ):
        self.mcp_clients = mcp_clients
        self.llm: Optional[ChatOpenAI] = None
        self.model = model
        self.system_prompt = system_prompt
        self.context = context

    async def init(self):
        """Initialize the agent and its MCP clients"""
        log_title('TOOLS')
        for client in self.mcp_clients:
            await client.init()
            
        # Get all tools from MCP clients
        tools = []
        for client in self.mcp_clients:
            tools.extend(client.get_tools())
            
        # Initialize LLM with tools
        self.llm = ChatOpenAI(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=tools,
            context=self.context
        )

    async def close(self):
        """Clean up resources"""
        for client in self.mcp_clients:
            await client.close()

    async def invoke(self, prompt: str):
        """Execute agent with given prompt"""
        if not self.llm:
            raise RuntimeError("Agent not initialized. Call init() first")
        
        return await self.llm.chat(prompt)