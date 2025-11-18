import os
import pytest
from unittest.mock import patch, MagicMock
from agent.rag.retrieval import load_documents, split_documents, create_vector_store, get_retriever

# Create dummy doc files for testing
@pytest.fixture(scope="module")
def create_dummy_docs():
    os.makedirs("docs", exist_ok=True)
    with open("docs/test_doc1.md", "w") as f:
        f.write("This is a test document.")
    with open("docs/test_doc2.md", "w") as f:
        f.write("This is another test document.")
    yield
    os.remove("docs/test_doc1.md")
    os.remove("docs/test_doc2.md")

def test_load_documents(create_dummy_docs):
    documents = load_documents("docs")
    assert len(documents) == 2
    assert "This is a test document." in documents[0].page_content or "This is another test document." in documents[0].page_content

def test_split_documents(create_dummy_docs):
    documents = load_documents("docs")
    chunks = split_documents(documents)
    assert len(chunks) > 0
    assert isinstance(chunks[0].page_content, str)

@patch("agent.rag.retrieval.SentenceTransformerEmbeddings")
@patch("agent.rag.retrieval.lancedb")
def test_create_vector_store(mock_lancedb, mock_embeddings, create_dummy_docs):
    mock_embeddings.return_value = MagicMock()
    mock_db = MagicMock()
    mock_lancedb.connect.return_value = mock_db
    mock_table = MagicMock()
    mock_db.create_table.return_value = mock_table
    
    documents = load_documents("docs")
    chunks = split_documents(documents)
    
    with patch("agent.rag.retrieval.LanceDB.from_documents") as mock_from_documents:
        vector_store = create_vector_store(chunks)
        mock_lancedb.connect.assert_called_with("your_project/lancedb")
        mock_db.create_table.assert_called()
        mock_from_documents.assert_called()
        assert vector_store is not None

@patch("agent.rag.retrieval.get_retriever")
def test_get_retriever(mock_get_retriever):
    mock_retriever = MagicMock()
    mock_get_retriever.return_value = mock_retriever
    
    retriever = get_retriever()
    
    assert retriever is not None
