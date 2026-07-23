from agent.memory.MemoryEngine.memory import Memory, MemoryItem
import sqlite3

class LongTermMemory(Memory):
    def __init__(self, db_path: str = "robot_memory.db"):
        self.con = sqlite3.connect(db_path)
        self.cur = self.con.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp TEXT NOT NULL
                    
            )
        """)
        self.con.commit()
    
    def get_all(self) -> list[MemoryItem]:
        data = self.cur.execute("SELECT * FROM memories").fetchall()
        return [MemoryItem(key=item[0], value=item[1], timestamp=item[2]) for item in data]
    
    def add(self, memory: MemoryItem) -> bool:
        return self.modify(memory)

    def modify(self, memory: MemoryItem) -> bool:
        try:
            self.cur.execute("""
                INSERT OR REPLACE INTO memories (key, value, timestamp)
                VALUES (?, ?, ?)
            """, (memory.key, memory.value, memory.timestamp))
            self.con.commit()
            return True
        except Exception as e:
            print(f"Failed: {e}")
            return False
        
    def get(self, key: str) -> MemoryItem | None:
        item = self.cur.execute("SELECT key, value, timestamp FROM memories WHERE key = ?", (key,)).fetchone()
        if item:
            return MemoryItem(key=item[0], value=item[1], timestamp=item[2])
        return None       
            
    def delete(self, key: str) -> bool:
        try:
            self.cur.execute("DELETE FROM memories WHERE key = ?", (key,))
            self.con.commit()
            if self.cur.rowcount == 0:
                return False
            return True
        except Exception as e:
            print(f"Failed: {e}")
            return False
    def close(self):
        self.cur.close()
        self.con.close()
