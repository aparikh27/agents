from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class MemoryItem:
    key: str
    value: str
    timestamp: float

class Memory(ABC):
    @abstractmethod
    def add(self, memory: MemoryItem) -> bool:
        pass
    @abstractmethod
    def get(self, key: str) -> MemoryItem | None:
        pass
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    @abstractmethod
    def modify(self, memory: MemoryItem) -> bool:
        pass