import asyncio
import os
from pathlib import Path

from .agent import Agent
from .mcp_client import MCPClient
from .embedding_retriever import EmbeddingRetriever
from .utils import log_title

# Constants
URL = "https://news.ycombinator.com/"
ROOT_PATH = Path(__file__).parent.parent

OUT_PATH = ROOT_PATH / "output"
TASK = f"""
告诉我Antonette的信息,先从我给你的context中找到相关信息,总结后创作一个关于她的故事
把故事和她的基本信息保存到{OUT_PATH}/antonette.md,输出一个漂亮md文件
"""

# Initialize MCP clients
fetch_mcp = MCPClient("mcp-server-fetch", "uvx", ["mcp-server-fetch"])
file_mcp = MCPClient(
    "mcp-server-file",
    "npx",
    ["-y", "@modelcontextprotocol/server-filesystem", str(OUT_PATH)]
)

async def main():
    # RAG retrieval
    context = await retrieve_context()
    
    # Initialize and run agent
    agent = Agent("openai/gpt-4", [fetch_mcp, file_mcp], context=context)
    await agent.init()
    await agent.invoke(TASK)
    await agent.close()

async def retrieve_context():
    """Retrieve relevant context using RAG"""
    embedding_retriever = EmbeddingRetriever("BAAI/bge-m3")
    
    # Load and embed knowledge files
    knowledge_dir = ROOT_PATH / "knowledge"
    for file in knowledge_dir.glob("*.md"):
        content = file.read_text()
        await embedding_retriever.embed_document(content)
        
    # Retrieve relevant context
    context = "\n".join(await embedding_retriever.retrieve(TASK, 3))
    
    log_title("CONTEXT")
    print(context)
    return context

if __name__ == "__main__":
    asyncio.run(main())