from agents.agent.memory_agent.MemoryEngine.long_term.long_mem import LongTermMemory
from agents.agent.memory_agent.MemoryEngine.short_term.short_mem import ShortTermMemory
from agents.agent.memory_agent.MemoryEngine.memory import Memory, MemoryItem

class MemoryManager(Memory):
    def __init__(self, capacity: int = 100, db_path: str = "robot_memory.db"):
        self.long = LongTermMemory(db_path)
        self.short = ShortTermMemory(capacity, self.long)

    def add(self, memory: MemoryItem) -> bool:
        return self.short.add(memory)

    def get(self, key: str) -> MemoryItem | None:
        get_short = self.short.get(key)
        if get_short is not None:
            return get_short
        return self.long.get(key)

    def delete(self, key: str) -> bool:
        del_short = self.short.delete(key)
        del_long = self.long.delete(key)
        return del_short or del_long

    def modify(self, memory: MemoryItem) -> bool:
        mod_short = self.short.modify(memory)
        if mod_short:
            return True
        return self.long.modify(memory)