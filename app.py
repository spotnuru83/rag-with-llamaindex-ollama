from llama_index.core import SimpleDirectoryReader

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# Load documents from a directory
documents = SimpleDirectoryReader(input_files=["./data/ATGL_ConferenceCall_2024.pdf","./data/motor_policy_terms_Conditions_PACKAGE_PRIVATECAR.pdf"]).load_data()

# create vector database file and a collection in which the embeddings will be stored
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("rag_collection")

# save embedding to disk
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#Create Ollama Embeddings 
from llama_index.embeddings.ollama import OllamaEmbedding

ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# Insert the documents in vector database indexed in form of ollama embeddings
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=ollama_embedding
)

print("Vector database is created with embeddings called the Vector Index!")

from llama_index.llms.ollama import Ollama
llm = Ollama(model="llama3.1", request_timeout=420.0)

# Query Data from the persisted index
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("can you give me cover for injuries that are covered. and please include the quote and page number in the document")
print(response)