import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("openai").disabled = True
logging.getLogger("httpx").disabled = True

class RAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        persist_directory = os.path.join(os.getcwd(), 'chroma_db')
        self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    async def upload_data(self, texts: List[str]):
        logger.info('Uploading info for RAG....')
        try:
            docs = self.text_splitter.create_documents(texts)
            await self.vectorstore.aadd_documents(docs)
        except Exception as e:
            logger.error(f"Error uploading data to RAG: {str(e)}")
            raise

async def store_information(rag: RAG, info: str):
    try:
        await rag.upload_data([info])
    except Exception as e:
        logger.error(f"Error storing information: {str(e)}")
        raise

async def get_knowledge(rag: RAG, query: str) -> str:
    try:
        docs = await rag.vectorstore.asimilarity_search(query)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error retrieving knowledge: {str(e)}")
        return f"Error retrieving knowledge: {str(e)}"