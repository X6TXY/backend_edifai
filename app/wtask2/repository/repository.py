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
        docs = self.vector_store.similarity_search(query=request, k=15)
        llm = OpenAI(temperature=0.8)
        prompt_template = CustomPromptTemplate()
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        evaluator_prompt = f"""
        firstly check number of words and punctuation,grammar mistakes.
        "Strict Essay Scoring Criteria for IELTS Evaluation"
The essay must be scored on a scale of 0.0 to 9.0.
If the essay response contains fewer than 100 words and 600 characters, the score should be 3.5.
The evaluation should be based on the IELTS criteria, focusing on four main bands: Task Response, Coherence and Cohesion, Lexical Resource, and Grammar.
Each band should be scored individually, with clear descriptions provided for each score point in Task Response, Coherence and Cohesion, Lexical Resource, and Grammar.
The total score for the essay should be calculated as the average of the scores in Task Response, Coherence and Cohesion, Lexical Resource, and Grammar.
The overall score should be rounded to the nearest 0.5.
In case the essay contains fewer than 50 words, the score should be 0. For other word count ranges, specific scores should be assigned accordingly.
The final score should be presented as a number within the range of 0.0 to 9.0, without any additional text or explanations.
The essay should be evaluated strictly based on the provided criteria for each band, focusing on addressing the task, coherence and organization, vocabulary usage, and grammar accuracy.
Any additional assessment or subjective comments should be avoided, and the evaluation should remain objective and accurate.
Example of essay scoring:

Task Response: 7.0
Coherence and Cohesion: 6.5
Lexical Resource: 7.5
Grammar: 7.0
Total score for the IELTS essay: (7.0 + 6.5 + 7.5 + 7.0) / 4 = 7.0
output ONLY SCORE OF ESSAY DON'T WRITE ANYTHING ELSE
        """

        with get_openai_callback() as cb:
            response = chain.run(
                input_documents=docs,
                question=evaluator_prompt,
                model="gpt-3.5-turbo-16k",
                temperature=0.8,
            )

        return response

    def get_answer(self, request: str):
        if self.vector_store is None:
            raise ValueError("Vector store is not loaded.")

        docs = self.vector_store.similarity_search(query=request, k=3)
        llm = OpenAI(temperature=0.8)
        prompt_template = CustomPromptTemplate()
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        evaluator_prompt = f"""Image You are IETLS examiner and you can only answer about IELTS.Be strict.Give feedback for each sections. Also give advices how i can improve this Essay.Don t give a overall score.And give advices which words i can use for improve essay sructure.This is Essay:{request}.Also output useful words for change repetion words ,check grammar mistakes, linking words. check essay for having punctuation and grammar mistakes.Also feedback and give an examples for task response, grammar, lexical resource also Coherence and Cohesion.In the end give useful vocabulary  \n\n"""

        with get_openai_callback() as cb:
            response = chain.run(
                input_documents=docs,
                question=evaluator_prompt,
                model="gpt-3.5-turbo-16k",
            )
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
