import os
import json
from typing import List
from tqdm import tqdm
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from VarMap import TEXT_FILE_ADDRESS, DATASET_ADDRESS

with open("System-Prompt.txt", "r") as file:
    QA_PAIRS_SYSTEM_PROMPT = file.read().replace("\n", " ")
with open("Human-Prompt.txt", "r") as file:
    QA_PAIRS_HUMAN_PROMPT = file.read().replace("\n", " ")

class QaPair(BaseModel):
    instruction: str = Field(description="系统指令")
    # "input": "人类输入",
    input: str = Field(description="人类输入（针对问题内容）")
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
    llm = ChatTongyi(model="qwen-max-latest")
    parser = JsonOutputParser(pydantic_object=QaPairs)
    chain = RunnableSequence(prompt, llm, parser)
    return chain


def main():
    chain = create_chain()
    documents = split_document(TEXT_FILE_ADDRESS)

    # for i, doc in enumerate(documents):
    #   print(f"Document chunk {i + 1}:")
    #   print(doc.page_content)
    #   print("-" * 80)

    with open(DATASET_ADDRESS, "a", encoding="utf-8") as f:
        for idx, doc in enumerate(tqdm(documents)):
            # print(f"Processing document chunk {idx + 1}")
            try:
                out = chain.invoke({"text": doc.page_content})
                # print(f"API response for chunk {idx + 1}: {out}")

                f.write(json.dumps(out, ensure_ascii=False, indent=2) + ",\n")
                f.flush()
            except:
                continue

if __name__ == "__main__":
    main()
