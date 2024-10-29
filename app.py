from llama_index.core import SimpleDirectoryReader

# Load documents from a directory
documents = SimpleDirectoryReader(input_files=["./data/ATGL_ConferenceCall_2024.pdf","./data/motor_policy_terms_Conditions_PACKAGE_PRIVATECAR.pdf"]).load_data()

from llama_index.embeddings.ollama import OllamaEmbedding

ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# create client
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("rag_collection")

# save embedding to disk
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=ollama_embedding
)

from llama_index.llms.ollama import Ollama
llm = Ollama(model="llama3.1", request_timeout=420.0)

# Query Data from the persisted index
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("What is the DEPRECIATION FOR FIXING IDV OF THE VEHICLE?")
print(response)