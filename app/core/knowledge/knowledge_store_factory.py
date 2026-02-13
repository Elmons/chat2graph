from threading import RLock

from app.core.common.system_env import SystemEnv
from app.core.common.type import KnowledgeStoreType
from app.core.knowledge.knowledge_store import KnowledgeStore


class KnowledgeStoreFactory:
    """Knowledge store factory."""
    _stores: dict[tuple[KnowledgeStoreType, str], KnowledgeStore] = {}
    _lock = RLock()

    @classmethod
    def get_or_create(cls, name: str) -> KnowledgeStore:
        """Get or create a cached knowledge store instance."""
        from app.plugin.dbgpt.dbgpt_knowledge_store import GraphKnowledgeStore, VectorKnowledgeStore

        store_type = SystemEnv.KNOWLEDGE_STORE_TYPE
        key = (store_type, str(name))
        with cls._lock:
            if key in cls._stores:
                return cls._stores[key]

            if store_type == KnowledgeStoreType.VECTOR:
                store = VectorKnowledgeStore(name)
            elif store_type == KnowledgeStoreType.GRAPH:
                store = GraphKnowledgeStore(name)
            else:
                raise ValueError(f"Cannot create knowledge store of type {store_type}")

            cls._stores[key] = store
            return store
