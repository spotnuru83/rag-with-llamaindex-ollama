import os
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_documents(input_files):
    documents = SimpleDirectoryReader(input_files=input_files).load_data()
    print("----Documents Loaded----")
    print(len(documents))
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    return document

def build_sentence_window_index(
    documents,
    llm,
    embed_model="BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    Settings.llm = llm 
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)
    Settings.node_parser = node_parser 
    
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, 
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2,model="BAAI/bge-reranker-base"
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model=model
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


def ask_query(inputs, q,temperature=0.1,store_to="./sentence_index",model="llama3.1"):

    document = get_documents(input_files=inputs)
    
    index = build_sentence_window_index(
    [document],
    llm=Ollama(model=model, temperature=temperature),
    save_dir=store_to,
    )

    query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)
    
    response = query_engine.query(q)

    return response
    

input_files=["./data/ATGL_ConferenceCall_2024.pdf","./data/motor_policy_terms_Conditions_PACKAGE_PRIVATECAR.pdf"]

response = ask_query(input_files,"Please give me summary of the conference call")
print(response)


