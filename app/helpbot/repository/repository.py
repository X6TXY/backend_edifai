import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader

load_dotenv()


class CustomPromptTemplate:
    def format(self, prompt):
        return f"Only ask about IELTS resources: {prompt}"


class HelpBotRepository:
    def __init__(self):
        self.vector_store = None

    def load_vector_store(self):
        pdf_path = "static/Helpbot.pdf"
        if os.path.exists(pdf_path):
            pdf_reader = PdfReader(pdf_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            store_name = os.path.splitext(os.path.basename(pdf_path))[0]
            vector_store_path = f"{store_name}.pkl"

            if os.path.exists(vector_store_path):
                with open(vector_store_path, "rb") as f:
                    self.vector_store = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                self.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                with open(vector_store_path, "wb") as f:
                    pickle.dump(self.vector_store, f)
        else:
            raise FileNotFoundError("PDF file not found.")

    def get_answer(self, request: str):
        if self.vector_store is None:
            raise ValueError("Vector store is not loaded.")

        docs = self.vector_store.similarity_search(query=request, k=3)
        llm = OpenAI(temperature=0.7)
        prompt_template = CustomPromptTemplate()
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        evaluator_prompt = f"""You must be answer my questions from pdf file.And you ask about IELTS.I need study:{request}.Give me information what should i do """

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=evaluator_prompt)

        return response
