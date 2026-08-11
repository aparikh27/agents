from agents.agent.memory_agent.MemoryEngine.long_term.long_mem import LongTermMemory
from agents.agent.memory_agent.MemoryEngine.memory import Memory, MemoryItem
from collections import deque

class ShortTermMemory(Memory):
    def __init__(self, capacity: int = 100, longterm: LongTermMemory = None):
        self.capacity = capacity
        self.short_term_memory: dict[str, MemoryItem] = {}
        self.addition_timeline = deque()
        self.long_term_memory = longterm

    def add(self, item: MemoryItem) -> bool:
        if self.modify(item):
            return True
        if len(self.short_term_memory) < self.capacity:
            self.short_term_memory[item.key] = item
            self.addition_timeline.append(item.key)
            return True
        else:
            last_item = self.addition_timeline.popleft()  # Remove the oldest entry
            if self.long_term_memory:
                self.long_term_memory.add(self.short_term_memory[last_item])  # Move it to long-term memory
            del self.short_term_memory[last_item]
            self.short_term_memory[item.key] = item
            self.addition_timeline.append(item.key)
            return True
  
    def delete(self, key: str) -> bool:
        if key in self.short_term_memory:
            del self.short_term_memory[key]
            self.addition_timeline.remove(key)
            return True
        return False

    def get(self, key: str) -> MemoryItem | None:
        return self.short_term_memory.get(key)

    def modify(self, item: MemoryItem) -> bool:
        if item.key in self.short_term_memory:
            self.short_term_memory[item.key] = item
            return True
        return False