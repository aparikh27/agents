import time

import pytest

from agents.agent.memory_agent.MemoryEngine.memory import MemoryItem
from agents.agent.memory_agent.MemoryEngine.short_term.short_mem import ShortTermMemory
from agents.agent.memory_agent.MemoryEngine.long_term.long_mem import LongTermMemory


###############################################################################
# Fixtures
###############################################################################

@pytest.fixture
def long_term(tmp_path):
    db = tmp_path / "test_memory.db"

    mem = LongTermMemory(db_path=str(db))

    yield mem

    mem.close()


@pytest.fixture
def short_term():
    return ShortTermMemory(capacity=3)


@pytest.fixture
def short_term_with_long(long_term):
    return ShortTermMemory(
        capacity=3,
        longterm=long_term
    )


###############################################################################
# Helper
###############################################################################

def create_item(key, value):
    return MemoryItem(
        key=key,
        value=value,
        timestamp=time.time()
    )


###############################################################################
# Long Term Tests
###############################################################################

def test_add_and_get(long_term):

    memory = create_item("cup", "blue")

    assert long_term.add(memory)

    result = long_term.get("cup")

    assert result is not None
    assert result.key == "cup"
    assert result.value == "blue"


def test_modify(long_term):

    long_term.add(create_item("cup", "blue"))

    long_term.modify(create_item("cup", "red"))

    result = long_term.get("cup")

    assert result.value == "red"


def test_delete(long_term):

    long_term.add(create_item("cup", "blue"))

    assert long_term.delete("cup")

    assert long_term.get("cup") is None


def test_delete_nonexistent(long_term):

    assert long_term.delete("does_not_exist") is False


def test_get_nonexistent(long_term):

    assert long_term.get("random") is None


def test_get_all(long_term):

    long_term.add(create_item("a", "1"))
    long_term.add(create_item("b", "2"))
    long_term.add(create_item("c", "3"))

    memories = long_term.get_all()

    assert len(memories) == 3

    keys = {m.key for m in memories}

    assert keys == {"a", "b", "c"}


###############################################################################
# Short Term Tests
###############################################################################

def test_short_term_add(short_term):

    short_term.add(create_item("apple", "fruit"))

    assert short_term.get("apple").value == "fruit"


def test_short_term_modify(short_term):

    short_term.add(create_item("apple", "fruit"))

    short_term.modify(create_item("apple", "green"))

    assert short_term.get("apple").value == "green"


def test_short_term_delete(short_term):

    short_term.add(create_item("apple", "fruit"))

    assert short_term.delete("apple")

    assert short_term.get("apple") is None


def test_duplicate_add(short_term):

    short_term.add(create_item("cup", "blue"))

    short_term.add(create_item("cup", "red"))

    assert len(short_term.short_term_memory) == 1

    assert short_term.get("cup").value == "red"


###############################################################################
# Eviction Tests
###############################################################################

def test_eviction_without_long_term():

    stm = ShortTermMemory(capacity=2)

    stm.add(create_item("a", "1"))
    stm.add(create_item("b", "2"))
    stm.add(create_item("c", "3"))

    assert stm.get("a") is None

    assert stm.get("b") is not None
    assert stm.get("c") is not None


def test_eviction_to_long_term(short_term_with_long):

    stm = short_term_with_long

    stm.add(create_item("a", "1"))
    stm.add(create_item("b", "2"))
    stm.add(create_item("c", "3"))
    stm.add(create_item("d", "4"))

    #
    # a should have been evicted
    #

    assert stm.get("a") is None

    recovered = stm.long_term_memory.get("a")

    assert recovered is not None
    assert recovered.value == "1"


###############################################################################
# Stress Test
###############################################################################

def test_many_insertions(long_term):

    stm = ShortTermMemory(
        capacity=50,
        longterm=long_term
    )

    for i in range(500):

        stm.add(
            create_item(
                f"key{i}",
                f"value{i}"
            )
        )

    #
    # 50 remain in RAM
    #

    assert len(stm.short_term_memory) == 50

    #
    # earliest should be in SQLite
    #

    first = long_term.get("key0")

    assert first is not None
    assert first.value == "value0"

    #
    # latest still in RAM
    #

    assert stm.get("key499") is not None


###############################################################################
# Persistence Test
###############################################################################

def test_database_persistence(tmp_path):

    db = tmp_path / "persist.db"

    mem1 = LongTermMemory(str(db))

    mem1.add(create_item("robot", "atlas"))

    mem1.close()

    mem2 = LongTermMemory(str(db))

    result = mem2.get("robot")

    assert result is not None
    assert result.value == "atlas"

    mem2.close()