from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import json

@dataclass
class Tool:
    name: str 
    description: str
    input_schema: Dict[str, Any]

class MCPClient:
    """Model Context Protocol Client implementation"""
    
    def __init__(self, name: str, command: str, args: List[str], version: str = "0.0.1"):
        self.name = name
        self.command = command
        self.args = args
        self.version = version
        self.transport = None
        self.tools: List[Tool] = []
        self.process: Optional[asyncio.subprocess.Process] = None
    
    async def init(self):
        """Initialize MCP client and connect to server"""
        await self.connect_to_server()
        
    async def close(self):
        """Clean up resources"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
    
    def get_tools(self) -> List[Tool]:
        """Get available tools"""
        return self.tools
        
    async def call_tool(self, name: str, params: Dict[str, Any]):
        """Call a tool with given parameters"""
        if not self.process:
            raise RuntimeError("MCP Client not connected")
            
        request = {
            "name": name,
            "arguments": params
        }
        
        # Write request to process stdin
        self.process.stdin.write(json.dumps(request).encode() + b"\n")
        await self.process.stdin.drain()
        
        # Read response from stdout
        response = await self.process.stdout.readline()
        return json.loads(response)
        
    async def connect_to_server(self):
        """Connect to MCP server process"""
        try:
            # Start server process
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE
            )
            
            # Get available tools
            tools_request = {"type": "list_tools"}
            self.process.stdin.write(json.dumps(tools_request).encode() + b"\n")
            await self.process.stdin.drain()
            
            tools_response = await self.process.stdout.readline()
            tools_data = json.loads(tools_response)
            
            self.tools = [
                Tool(
                    name=t["name"],
                    description=t["description"], 
                    input_schema=t["input_schema"]
                )
                for t in tools_data["tools"]
            ]
            
            print(f"Connected to server with tools: {[t.name for t in self.tools]}")
            
        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            raise