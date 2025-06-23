#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内存管理模块
负责管理Agent的会话内存，包括创建、存储、检索和回滚等功能
"""

import os
import json
import time
import logging

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field, validator


class MemoryConfig(BaseModel):
    """内存管理器配置"""
    
    # 存储配置
    storage_dir: str = "data/memory"
    session_file_format: str = "{agent_id}.json"
    
    # 内存配置
    max_memory_items: int = Field(default=100, description="每个会话最大内存条目数")
    enable_compression: bool = Field(default=False, description="是否启用压缩存储")
    
    # 回滚配置
    max_history_states: int = Field(default=10, description="保存的最大历史状态数")
    auto_save_interval: int = Field(default=5, description="自动保存间隔(步骤数)")


class MemoryItem(BaseModel):
    """内存条目"""
    
    # 元数据
    id: str                               # 唯一标识符
    timestamp: float                      # 创建时间戳
    type: str                             # 条目类型
    
    # 内容
    content: Dict[str, Any]               # 条目内容
    
    # 关系
    parent_id: Optional[str] = None       # 父条目ID
    references: List[str] = Field(default_factory=list)  # 引用的其他条目ID
    
    # 标签和元数据
    tags: List[str] = Field(default_factory=list)        # 标签
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 元数据
    
    class Config:
        arbitrary_types_allowed = True


class MemorySession(BaseModel):
    """内存会话"""
    
    # 会话信息
    agent_id: str                         # Agent ID
    session_id: str                       # 会话ID
    created_at: float                     # 创建时间戳
    last_updated: float                   # 最后更新时间戳
    
    # 内存内容
    items: Dict[str, MemoryItem] = Field(default_factory=dict)  # 内存条目
    
    # 会话状态
    history_states: List[Dict[str, Any]] = Field(default_factory=list)  # 历史状态
    current_state_idx: int = -1           # 当前状态索引
    
    # 统计信息
    total_items: int = 0                  # 总条目数
    
    class Config:
        arbitrary_types_allowed = True


class MemoryManager:
    """
    内存管理器
    负责管理Agent的会话内存
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        初始化内存管理器
        
        Args:
            config: 内存配置
        """
        self.config = config or MemoryConfig()
        self.sessions: Dict[str, MemorySession] = {}
        self.logger = self._setup_logger()
        
        # 创建存储目录
        os.makedirs(self.config.storage_dir, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("memory_manager")
        logger.setLevel(logging.INFO)
        
        # 添加控制台处理器
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_session_file_path(self, agent_id: str) -> str:
        """获取会话文件路径"""
        file_name = self.config.session_file_format.format(agent_id=agent_id)
        return os.path.join(self.config.storage_dir, file_name)
    
    def init_session(self, agent_id: str, initial_data: Any) -> None:
        """
        初始化会话
        
        Args:
            agent_id: Agent ID
            initial_data: 初始数据
        """
        # 检查会话是否已存在
        if agent_id in self.sessions:
            self.logger.warning(f"会话 {agent_id} 已存在，将被重置")
        
        # 创建新会话
        current_time = time.time()
        session = MemorySession(
            agent_id=agent_id,
            session_id=f"{agent_id}_{int(current_time)}",
            created_at=current_time,
            last_updated=current_time
        )
        
        # 添加初始数据
        initial_item = MemoryItem(
            id=f"item_{int(current_time)}",
            timestamp=current_time,
            type="initial_input",
            content={"data": initial_data}
        )
        
        session.items[initial_item.id] = initial_item
        session.total_items += 1
        
        # 保存初始状态
        self._save_state(session)
        
        # 保存会话
        self.sessions[agent_id] = session
        self.logger.info(f"会话 {agent_id} 已初始化")
    
    def save_session(self, agent_id: str) -> bool:
        """
        保存会话到文件
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功
        """
        if agent_id not in self.sessions:
            self.logger.error(f"会话 {agent_id} 不存在")
            return False
        
        session = self.sessions[agent_id]
        session.last_updated = time.time()
        
        try:
            file_path = self._get_session_file_path(agent_id)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session.dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"会话 {agent_id} 已保存到 {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存会话 {agent_id} 失败: {e}")
            return False
    
    def load_session(self, agent_id: str) -> bool:
        """
        从文件加载会话
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功
        """
        file_path = self._get_session_file_path(agent_id)
        
        if not os.path.exists(file_path):
            self.logger.error(f"会话文件 {file_path} 不存在")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                session_dict = json.load(f)
            
            session = MemorySession.parse_obj(session_dict)
            self.sessions[agent_id] = session
            
            self.logger.info(f"会话 {agent_id} 已从 {file_path} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载会话 {agent_id} 失败: {e}")
            return False
    
    def _save_state(self, session: MemorySession) -> None:
        """
        保存当前状态到历史
        
        Args:
            session: 会话对象
        """
        # 创建当前状态的快照
        items_snapshot = {k: v.dict() for k, v in session.items.items()}
        state = {
            "timestamp": time.time(),
            "items": items_snapshot,
            "total_items": session.total_items
        }
        
        # 移除当前索引之后的所有状态(如果曾经回滚过)
        if session.current_state_idx < len(session.history_states) - 1:
            session.history_states = session.history_states[:session.current_state_idx + 1]
        
        # 添加当前状态
        session.history_states.append(state)
        session.current_state_idx = len(session.history_states) - 1
        
        # 如果历史状态超过限制，移除最早的状态
        if len(session.history_states) > self.config.max_history_states:
            session.history_states = session.history_states[-self.config.max_history_states:]
            session.current_state_idx = len(session.history_states) - 1
    
    def can_rollback(self, agent_id: str) -> bool:
        """
        检查是否可以回滚
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否可以回滚
        """
        if agent_id not in self.sessions:
            return False
        
        session = self.sessions[agent_id]
        return session.current_state_idx > 0
    
    def rollback(self, agent_id: str) -> bool:
        """
        回滚到上一个状态
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功
        """
        if not self.can_rollback(agent_id):
            self.logger.warning(f"会话 {agent_id} 无法回滚")
            return False
        
        session = self.sessions[agent_id]
        session.current_state_idx -= 1
        
        # 加载上一个状态
        previous_state = session.history_states[session.current_state_idx]
        
        # 恢复状态
        items_dict = previous_state["items"]
        session.items = {k: MemoryItem.parse_obj(v) for k, v in items_dict.items()}
        session.total_items = previous_state["total_items"]
        
        self.logger.info(f"会话 {agent_id} 已回滚到状态 {session.current_state_idx}")
        return True
    
    def add_item(self, agent_id: str, item_type: str, content: Dict[str, Any], 
                 parent_id: Optional[str] = None, tags: Optional[List[str]] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        添加内存条目
        
        Args:
            agent_id: Agent ID
            item_type: 条目类型
            content: 条目内容
            parent_id: 父条目ID
            tags: 标签
            metadata: 元数据
            
        Returns:
            条目ID，如果失败则返回None
        """
        if agent_id not in self.sessions:
            self.logger.error(f"会话 {agent_id} 不存在")
            return None
        
        session = self.sessions[agent_id]
        
        # 检查是否超出内存限制
        if session.total_items >= self.config.max_memory_items:
            self.logger.warning(f"会话 {agent_id} 已达到最大内存条目数限制")
            # 可以实现内存清理策略，例如移除最旧的条目
        
        # 创建内存条目
        item_id = f"item_{int(time.time())}_{session.total_items}"
        item = MemoryItem(
            id=item_id,
            timestamp=time.time(),
            type=item_type,
            content=content,
            parent_id=parent_id,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # 添加到会话
        session.items[item_id] = item
        session.total_items += 1
        
        # 每隔一定步骤自动保存状态
        if session.total_items % self.config.auto_save_interval == 0:
            self._save_state(session)
        
        return item_id
    
    def get_item(self, agent_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        """
        获取内存条目
        
        Args:
            agent_id: Agent ID
            item_id: 条目ID
            
        Returns:
            条目内容，如果不存在则返回None
        """
        if agent_id not in self.sessions:
            self.logger.error(f"会话 {agent_id} 不存在")
            return None
        
        session = self.sessions[agent_id]
        
        if item_id not in session.items:
            self.logger.warning(f"条目 {item_id} 不存在")
            return None
        
        return session.items[item_id].dict()
    
    def update_item(self, agent_id: str, item_id: str, content: Dict[str, Any]) -> bool:
        """
        更新内存条目
        
        Args:
            agent_id: Agent ID
            item_id: 条目ID
            content: 新内容
            
        Returns:
            是否成功
        """
        if agent_id not in self.sessions:
            self.logger.error(f"会话 {agent_id} 不存在")
            return False
        
        session = self.sessions[agent_id]
        
        if item_id not in session.items:
            self.logger.warning(f"条目 {item_id} 不存在")
            return False
        
        # 更新内容
        session.items[item_id].content.update(content)
        session.items[item_id].timestamp = time.time()
        
        return True
    
    def get_session_data(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话数据
        
        Args:
            agent_id: Agent ID
            
        Returns:
            会话数据，如果不存在则返回None
        """
        if agent_id not in self.sessions:
            # 尝试从文件加载
            if not self.load_session(agent_id):
                return None
        
        return self.sessions[agent_id].dict()
    
    def search_items(self, agent_id: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        搜索内存条目
        
        Args:
            agent_id: Agent ID
            query: 查询条件，例如 {"type": "observation", "tags": ["important"]}
            
        Returns:
            匹配的条目列表
        """
        if agent_id not in self.sessions:
            self.logger.error(f"会话 {agent_id} 不存在")
            return []
        
        session = self.sessions[agent_id]
        results = []
        
        for item in session.items.values():
            match = True
            
            for k, v in query.items():
                if k == "tags":
                    # 检查是否包含所有指定标签
                    if not all(tag in item.tags for tag in v):
                        match = False
                        break
                elif hasattr(item, k):
                    # 检查普通字段
                    if getattr(item, k) != v:
                        match = False
                        break
                elif k == "content" and isinstance(v, dict):
                    # 检查内容中的特定字段
                    for ck, cv in v.items():
                        if ck not in item.content or item.content[ck] != cv:
                            match = False
                            break
            
            if match:
                results.append(item.dict())
        
        return results
    
    def get_items_by_type(self, agent_id: str, item_type: str) -> List[Dict[str, Any]]:
        """
        获取指定类型的所有条目
        
        Args:
            agent_id: Agent ID
            item_type: 条目类型
            
        Returns:
            条目列表
        """
        return self.search_items(agent_id, {"type": item_type})
    
    def clear_session(self, agent_id: str) -> bool:
        """
        清除会话
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功
        """
        if agent_id not in self.sessions:
            self.logger.warning(f"会话 {agent_id} 不存在")
            return False
        
        # 删除会话
        del self.sessions[agent_id]
        
        # 删除会话文件
        file_path = self._get_session_file_path(agent_id)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                self.logger.info(f"会话文件 {file_path} 已删除")
            except Exception as e:
                self.logger.error(f"删除会话文件 {file_path} 失败: {e}")
                return False
        
        self.logger.info(f"会话 {agent_id} 已清除")
        return True
    
    def get_all_sessions(self) -> List[str]:
        """
        获取所有会话ID
        
        Returns:
            会话ID列表
        """
        # 合并内存中的会话和存储目录中的会话
        memory_sessions = set(self.sessions.keys())
        
        # 从存储目录获取会话
        stored_sessions = set()
        if os.path.exists(self.config.storage_dir):
            for file_name in os.listdir(self.config.storage_dir):
                if file_name.endswith(".json"):
                    agent_id = file_name.rsplit(".", 1)[0]
                    stored_sessions.add(agent_id)
        
        return list(memory_sessions.union(stored_sessions)) 
