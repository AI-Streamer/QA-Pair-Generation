import os
import json
from typing import List
from tqdm import tqdm
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def split_document(filepath):
    loader = UnstructuredFileLoader(filepath)
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    documents = loader.load_and_split(text_spliter)
    return documents
