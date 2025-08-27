import os
import json
from typing import List
from tqdm import tqdm
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )


class QaPair(BaseModel):
    instruction: str = Field(description="问题内容")
    # "input": "人类输入（选填）",
    input: str = Field(description="人类输入（针对问题内容,选填）")
    output: str = Field(description="问题的回答")


class QaPairs(BaseModel):
    qas: List[QaPair] = Field(description="问答对列表")


def split_document(filepath):
    loader = UnstructuredFileLoader(filepath)
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    documents = loader.load_and_split(text_spliter)
    return documents


def create_chain():
    prompt = ChatPromptTemplate.from_messages(
        [("system", QA_PAIRS_SYSTEM_PROMPT), ("human", QA_PAIRS_HUMAN_PROMPT)]
    )
    llm = ChatTongyi(model="qwen-plus-latest").with_structured_output(
        method="json_mode"
    )
    parser = JsonOutputParser(pydantic_object=QaPairs)
    chain = RunnableSequence([prompt, llm, parser])
    return chain


def main():
    chain = create_chain()
    documents = split_document("data/12.txt")

    # for i, doc in enumerate(documents):
    #   print(f"Document chunk {i + 1}:")
    #   print(doc.page_content)
    #   print("-" * 80)

    with open("dataset.json", "a", encoding="utf-8") as f:
        bar = tqdm(total=len(documents))
        for idx, doc in enumerate(documents):
            print(doc.page_content)

            print(f"Processing document chunk {idx + 1}")
            out = chain.invoke({"text": doc.page_content})
            print(f"API response for chunk {idx + 1}: {out}")

            f.write(json.dumps(out, ensure_ascii=False, indent=2) + ",\n")
            f.flush()
            bar.update(1)
            bar.close()
