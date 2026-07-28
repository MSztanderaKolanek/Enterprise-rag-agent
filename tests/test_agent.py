import pytest
from main import KnowledgeAgent

def test_agent_initialization():
    agent = KnowledgeAgent(chunk_size=200)
    assert agent.text_splitter._chunk_size == 200

def test_query_without_ingestion_raises_error():
    agent = KnowledgeAgent()
    with pytest.raises(ValueError):
        agent.query("What is RAG?")
