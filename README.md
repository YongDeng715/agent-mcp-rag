# agent-mcp-rag
Augmented LLM Chatbot Agent (Chat + MCP + RAG) from scratch in Python

## Introduction

- **Augmented LLM** (Chat + MCP + RAG)
- 不依赖框架
    - LangChain, LlamaIndex, CrewAI, AutoGen
- RAG 极简版和 改进版CRAG (Corrective RAG)
    - RAG 极简版: 从知识中检索出有关信息，注入到上下文
    - CRAG: 
- 任务：
    - 阅读网页 → 整理一份总结 → 保存到文件
    - 本地文档 → 查询相关资料 → 注入上下文

## Architecture

## User Guide

```bash
git clone git@github.com:YongDeng715/agent-mcp-rag.git
cd agent-mcp-rag
```

## LLM support

## About MCP

- [MCP architecture](https://modelcontextprotocol.io/docs/concepts/architecture)
- [MCP client](https://modelcontextprotocol.io/quickstart/client)
- [fetch MCP](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch)
- [Filesystem MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch)
- [fastmcp tutorials](https://github.com/jlowin/fastmcp/tree/main/docs/tutorials)

## About RAG

- [Retrieval Augmented Generation](https://scriv.ai/guides/retrieval-augmented-generation-overview/)
    - [RAG 译文](https://www.yuque.com/serviceup/misc/cn-retrieval-augmented-generation-overview)
- [Corrective RAG](https://arxiv.org/pdf/2401.15884)
    - [CRAG by scratch](https://github.com/FareedKhan-dev/all-rag-techniques/blob/main/20_crag.ipynb)
    - [CRAG by scratch-zh](https://github.com/liu673/rag-all-techniques/blob/master/src/full/20_crag.ipynb)
    - [CRAG by LangGraph](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_crag.ipynb)

## FAQ

Please refer to the [FAQ.md] for more details.

## Acknowledgments

We would like to extend our sincere appreciation to the following projects for their invaluable contributions:
- [llm-mcp-rag](https://github.com/KelvinQiu802/llm-mcp-rag)