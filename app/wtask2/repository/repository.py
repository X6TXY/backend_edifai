import os
import pickle
from collections import Counter
from datetime import datetime
from typing import Optional

from bson.objectid import ObjectId
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from pymongo.cursor import Cursor
from pymongo.database import Database
from PyPDF2 import PdfReader

load_dotenv()


class CustomPromptTemplate:
    def format(self, prompt):
        return f"Only ask about IELTS: {prompt}"


class Wtask2Repository:
    def __init__(self, database: Database):
        self.vector_store = None
        self.database = database
        self.user = None

    def load_vector_store(self):
        pdf_path = "static/Document.pdf"
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

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        user = self.database["users"].find_one(
            {
                "_id": ObjectId(user_id),
            }
        )
        self.user = user
        return self.user

    def get_score(self, request: str):
        if self.vector_store is None:
            raise ValueError("Vector store is not loaded.")
        docs = self.vector_store.similarity_search(query=request, k=3)
        llm = OpenAI(temperature=0.8)
        prompt_template = CustomPromptTemplate()
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        evaluator_prompt = f"""Image you are IELTS writing task 2 examiner.Maximum score 9.0 and minimum score 0.0.Output only mark without feedback and other sentences.\nFamiliarize yourself with the marking criteria: Before you start using the IELTS writing checker, it s important to understand the criteria that the examiners use to mark your writing. This will help you understand what you need to focus on to improve your score.
Practice writing regularly: To get the most out of the IELTS writing checker, it s important to practice writing regularly. This will help you improve your writing skills and give you more opportunities to use the checker.
Analyze your mistakes: When the writing checker highlights your mistakes, take the time to analyze them and understand why you made them. This will help you avoid making the same mistakes in the future.
Use the feedback to improve your writing: The IELTS writing checker provides feedback on your writing, so use it to your advantage. Take note of the areas where you need to improve and make the necessary changes to your writing.
Work on your time management: During the IELTS exam, time management is crucial. To prepare for this, try to complete your writing tasks within the allotted time and use the writing checker to check your work quickly.
Don t rely on the IELTS writing checker entirely: While the writing checker is a useful tool, it s important to remember that it s not perfect. Use it as a guide, but don t rely on it entirely. Always use your own judgement and common sense when it comes to your writing.Accurate mark of this essay:"""

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=evaluator_prompt)

        return response

    def get_answer(self, request: str):
        if self.vector_store is None:
            raise ValueError("Vector store is not loaded.")

        docs = self.vector_store.similarity_search(query=request, k=3)
        llm = OpenAI(temperature=0.8)
        prompt_template = CustomPromptTemplate()
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        evaluator_prompt = f"""Image You are IETLS examiner and you can only answer about IELTS.Be strict.Give feedback for each sections. Also give advices how i can improve this Essay.Don t give a overall score.And give advices which words i can use for improve essay sructure.This is Essay:{request}.\n\n"""

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=evaluator_prompt)
        payload = {
            "user_id": self.user["_id"],
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "response": response,
            "request": request,
            "score": self.get_score(
                request
            ),  # Calculate the score and add it to the payload
        }
        self.database["response"].insert_one(payload)
        return response

    def get_dates(self) -> dict[str, int]:
        response_collection = self.database["response"]
        responses = list(response_collection.find({"user_id": self.user["_id"]}))
        daily_submissions = Counter()
        for response in responses:
            date = response["date"].split("T")[0]  # Extract only the date portion
            daily_submissions[date] += 1
        return dict(daily_submissions)

    def get_responses_by_user_id(self, user_id: str) -> Cursor:
        return (
            self.database["response"]
            .find({"user_id": ObjectId(user_id)})
            .sort("_id", -1)
        )
