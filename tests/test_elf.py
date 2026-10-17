from elfhub import Elf


def test_elf_responds():
    agent = Elf()
    assert agent.invoke() == "Hi !"


def test_elf_callable():
    assert Elf()() == "Hi !"
