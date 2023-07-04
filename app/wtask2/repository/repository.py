from datetime import datetime
from typing import Any, Optional

from bson.objectid import ObjectId
from pymongo.database import Database
from pymongo.results import DeleteResult, UpdateResult

import os
import pickle
import streamlit as st
from annotated_text import annotated_text
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space

load_dotenv()



class WritingTask2:
    def __init__(self, database: Database):
        self.database = database
    
