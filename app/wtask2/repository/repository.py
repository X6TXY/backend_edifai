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
       !IMPORTANT Output only score without Band or The overall score of the essay is.

       1.Firstly check Grammar by this criterias mark essay:
       Band 1.0: No sentence structure, constant grammar and punctuation errors, unclear communication. 
       Band 2.0: Basic sentence structures, frequent grammar and punctuation errors, communication often unclear. 
       Band 3.0: Some simple sentence structures, many grammar and punctuation errors, communication somewhat clear. 
       Band 4.0: Fair use of simple sentences, several grammar and punctuation errors, communication generally clear.
       Band 5.0: Limited sentence structures, frequent grammar errors, punctuation issues, causing reader difficulty. 
       Band 6.0: Uses simple and complex sentences, some grammar and punctuation errors, clear communication. 
       Band 7.0: Variety of complex sentences, mostly error-free, good grammar control, occasional errors. 
       Band 8.0: Variety of complex sentences, mostly error-free. 
       Band 9.0: Expert use of complex sentences, nearly all sentences are error-free, perfect grammar control, flawless punctuation, sophisticated communication.
       2.Secondly check Lexical Resource.And mark by this criterias:
       Band 1.0: No vocabulary range, constant spelling errors, unclear communication. 
       Band 2.0: Limited vocabulary, frequent spelling errors, unclear communication. 
       Band 3.0: Basic vocabulary range, many spelling errors, communication somewhat clear but often hindered by word formation errors.
       Band 4.0: Adequate vocabulary, some spelling errors, clear communication. 
       Band 5.0: Limited vocabulary, frequent spelling errors, causing reader difficulty. 
       Band 6.0: Adequate vocabulary, some uncommon words with errors, clear communication. 
       Band 7.0: Sufficient vocabulary, uses less common words, aware of style, occasional errors. 
       Band 8.0: Wide vocabulary, skillful use of uncommon words, rare errors. Improve by focusing on collocations, careful paraphrasing, topic-appropriate words, correct spelling, avoiding errors and informal language.
       Band 9.0: Wide vocabulary range, rare spelling errors, precise use of less common words, sophisticated communication.
       3.Thirty check Coherence and Cohesion by this criterias:
       Band 1.0: No organization, no use of paragraphs or linking devices, no referencing. 
       Band 2.0: Minimal organization, improper use of paragraphs and linking devices, unclear referencing. 
       Band 3.0: Some organization, inconsistent use of paragraphs and linking devices, some referencing. 
       Band 4.0: Fair organization, uses paragraphs and linking devices with some errors, unclear referencing. 
       Band 5.0: Some organization, improper use of paragraphs and linking devices, may be repetitive. 
       Band 6.0: Coherent organization, uses paragraphs and linking devices with some errors, unclear referencing.
       Band 7.0: Logical organization, good paragraphing, varied linking devices, good referencing. 
       Band 8.0: Logical organization, appropriate paragraphing, excellent linking and cohesion, flawless referencing.
       Band 9.0:Excellent organization, perfect paragraphing, skillful use of linking devices, flawless referencing.
       4.Fourthly check Task Response by this criterias:
       Band 1.0: Fails to address task, no clear position, no conclusion, irrelevant ideas. 
       Band 2.0: Barely addresses task, unclear position, no proper conclusion, mostly irrelevant ideas. 
       Band 3.0: Partially addresses task, vague position, weak conclusion, some relevant ideas. 
       Band 4.0: Mostly addresses task, somewhat clear position, basic conclusion, relevant but underdeveloped ideas. 
       Band 5.0: Partial task address, unclear position, possibly no conclusion, underdeveloped ideas, irrelevant details. 
       Band 6.0: Addresses task, clear position, conclusion may be unclear, relevant but underdeveloped ideas. 
       Band 7.0: Fully addresses task, clear position throughout, relevant and developed ideas, possible lack of focus. 
       Band 8.0: Fully addresses task, clear position, relevant and well-developed ideas.
       Band 9.0: Fully addresses all parts of task, clear and consistent position, relevant and highly developed ideas, strong conclusion.
       5.After checking parts put overall mark.Calculate by summing scores and dividing by 4.The overall score is then rounded to the nearest whole or half band. 
       6.Output only a number of overall score
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

        evaluator_prompt = f"""Image You are IETLS examiner and you can only answer about IELTS.Be strict.Give feedback for each sections. Also give advices how i can improve this Essay.Don t give a overall score.And give advices which words i can use for improve essay sructure.This is Essay:{request}.
        Also output useful words for change repetion words ,check grammar mistakes, linking words. check essay for having punctuation and grammar mistakes.Also feedback and give an examples for task response, grammar, lexical resource also Coherence and Cohesion.
        In the end give useful vocabulary and give an examples where we can use this words.highlight essay key words \n\n"""

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
