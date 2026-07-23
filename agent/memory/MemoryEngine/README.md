# Memory Engine

> A hierarchical memory system for AI that provides both short-term and long-term memory for robotic perception, planning, and reasoning.

---

# Overview

The Memory Engine is responsible for storing, retrieving, updating, and managing information learned by AI.

Rather than relying on a single storage mechanism, the Memory Engine is composed of two layers:

- **Short-Term Memory (STM)** – Fast, in-memory storage for recent information.
- **Long-Term Memory (LTM)** – Persistent SQLite database for information that should survive cache eviction and program restarts.

This hierarchy mimics how humans utilize working memory and long-term memory while remaining lightweight and efficient enough for real-time robotics applications.

---

# Design Goals

The Memory Engine was designed with the following objectives:

- Fast retrieval of recently observed information
- Persistent storage across robot restarts
- Automatic movement of old memories into permanent storage
- Simple interface for other AI modules
- Easily extensible for future semantic memory systems

---

# High-Level Architecture

```text
                    Memory Manager
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
  Short-Term Memory              Long-Term Memory
      (Dictionary)                  (SQLite DB)
          │                               │
          └──────────Eviction────────────►│
```

The remainder of AI communicates with a **Memory Manager**, which delegates storage and retrieval to the appropriate memory system.

---

# Memory Flow

```text
          New Memory
               │
               ▼
      Short-Term Memory
               │
       Capacity Reached?
         │           │
        No          Yes
         │           │
         ▼           ▼
      Store      Evict Oldest
                     │
                     ▼
            Long-Term Memory
```

New information is always written into short-term memory first.

When short-term memory reaches capacity:

1. The oldest memory is removed.
2. The removed memory is inserted into long-term memory.
3. The new memory is inserted into short-term memory.

---

# Components

## MemoryItem

Every memory is represented by a `MemoryItem`.

```python
@dataclass
class MemoryItem:
    key: str
    value: str
    timestamp: float
```

### Fields

| Field | Description |
|--------|-------------|
| key | Unique identifier |
| value | Memory contents |
| timestamp | Unix timestamp indicating creation or modification |

Example:

```python
MemoryItem(
    key="user_name",
    value="Arav",
    timestamp=1719963200
)
```

---

## Memory Interface

Both memory implementations inherit from the same abstract interface.

```text
Memory
│
├── add()
├── get()
├── modify()
└── delete()
```

This ensures both implementations behave identically from the perspective of the rest of the system.

---

# Short-Term Memory

## Purpose

Short-term memory stores recently observed information in RAM for fast access.

Examples include:

- Current detected objects
- Recent speech commands
- Active goals
- Temporary planner state

---

## Data Structures

```text
Dictionary
```

Stores the actual memories.

```python
{
    key -> MemoryItem
}
```

```text
Deque
```

Maintains insertion order.

```text
Front -------------------- Back

Oldest                Newest
```

The deque allows O(1) eviction of the oldest memory.

---

## Internal Architecture

```text
                add()

                  │

                  ▼

         Already Exists?

          │          │

        Yes         No

          │          │

          ▼          ▼

      Update      Capacity Full?

                     │      │

                    No     Yes

                     │      │

                     ▼      ▼

                  Insert   Evict Oldest
                               │
                               ▼
                     Send to Long-Term
```

---

## Storage Complexity

Dictionary operations

| Operation | Complexity |
|-----------|------------|
| Add | O(1) |
| Get | O(1) |
| Modify | O(1) |
| Delete | O(1) |

Eviction

Deque pop from front

```
O(1)
```

---

# Long-Term Memory

## Purpose

Long-term memory permanently stores information that should survive cache eviction and program restarts.

Unlike short-term memory, information is stored inside a SQLite database.

---

## Why SQLite?

SQLite was selected because it provides:

- Persistent storage
- Zero server setup
- Built into Python
- ACID transactions
- Lightweight deployment
- Fast indexed lookup

No external database server is required.

---

# Database Schema

```
memories
```

| Column | Type |
|---------|------|
| key | TEXT PRIMARY KEY |
| value | TEXT |
| timestamp | TEXT |

---

# Database Architecture

```text
LongTermMemory

      │

sqlite3

      │

robot_memory.db
```

Every operation is translated into SQL.

| Method | SQL |
|---------|-----|
| add | INSERT OR REPLACE |
| get | SELECT |
| modify | INSERT OR REPLACE |
| delete | DELETE |

---

# Retrieval Flow

```text
get(key)

     │

     ▼

SQLite Database

     │

Memory Found?

   │        │

  Yes      No

   │        │

   ▼        ▼

MemoryItem  None
```

---

# Memory Lifecycle

```text
Observed

   │

   ▼

Short-Term Memory

   │

Recently Used

   │

Capacity Reached

   │

Evicted

   │

SQLite

   │

Retrieved Later
```

---

# Public API

## add()

Adds a new memory.

```python
memory.add(item)
```

Returns

```
True
```

---

## get()

Retrieves a memory.

```python
memory.get("robot_name")
```

Returns

```python
MemoryItem(...)
```

or

```python
None
```

---

## modify()

Updates an existing memory.

```python
memory.modify(item)
```

---

## delete()

Deletes a memory.

```python
memory.delete("robot_name")
```

---

## get_all()

Returns every stored memory.

Useful for

- debugging
- visualization
- dashboard integration

---

# Integration with BAI

The Memory Engine is intended to be accessed through a `MemoryManager`.

```text
Vision Module
      │
Speech Module
      │
Planner
      │
Executor
      │
      ▼
 MemoryManager
      │
      ├────────► Short-Term Memory
      │
      └────────► Long-Term Memory
```

Other modules should **never** directly access SQLite or the short-term cache.

Instead, they interact only with the Memory Manager.

Example:

```python
memory_manager.add(memory)
```

The manager determines where the memory should be stored.

---

# Design Decisions

## Why two memory systems?

Using only RAM would lose all information when the program exits.

Using only SQLite would significantly slow down frequent memory accesses.

A hybrid approach provides:

- fast access
- persistence
- scalability

---

## Why FIFO eviction?

A queue-based eviction policy was selected because it is:

- deterministic
- lightweight
- O(1)
- appropriate for the MVP

Future versions may implement:

- LRU
- LFU
- importance-based eviction
- semantic relevance scoring

---

## Why SQLite instead of a vector database?

The current objective is reliable storage rather than semantic retrieval.

SQLite provides:

- simpler deployment
- lower overhead
- easier debugging

Future releases may integrate:

- ChromaDB
- FAISS
- Pinecone

to enable semantic memory retrieval.

---

# Testing

The Memory Engine is validated using **pytest**.

Tests include:

- Memory insertion
- Retrieval
- Modification
- Deletion
- Cache eviction
- SQLite persistence
- Duplicate keys
- Stress testing
- Long-term memory transfer
- Database persistence across sessions

---

# Future Improvements

## Importance Scores

```text
Importance = 1–10
```

Important memories remain longer before eviction.

---

## Memory Categories

```text
Object

User

Task

Conversation

Location
```

---

## Semantic Search

Instead of

```
memory.get("bottle")
```

the robot could answer

> "Where did I leave my drink?"

using vector similarity search.

---

## Memory Promotion

Rather than promoting every evicted memory, only meaningful memories will be persisted.

Examples:

- User preferences
- Learned object ownership
- Successful tasks
- Environmental knowledge

---

## Forgetting

Low-value memories may eventually be removed after long periods of inactivity.

---

# Project Structure

```text
memory/
│
├── memory.py
├── short_mem.py
├── long_mem.py
├── memory_manager.py
└── robot_memory.db
```

---

# Summary

The Memory Engine provides AO with a hierarchical memory architecture consisting of a fast in-memory cache and a persistent SQLite-backed storage system. This design enables efficient real-time operation while preserving knowledge across executions, and serves as the foundation for future semantic memory and autonomous reasoning capabilities.