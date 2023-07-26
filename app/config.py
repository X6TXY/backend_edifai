from typing import Any

from pydantic import BaseSettings
from pymongo import MongoClient


class Config(BaseSettings):
    CORS_ORIGINS: list[str] = ["*"]
    CORS_HEADERS: list[str] = ["*"]
    CORS_METHODS: list[str] = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

    MONGOHOST: str = "localhost"
    MONGOPORT: str = "27017"
    MONGOUSER: str = "root"
    MONGOPASSWORD: str = "password"
    MONGODATABASE: str = "fastapi"
    MONGO_URL: str = "mongodb+srv://x6txy:baha2710@cluster0.7ohnbpk.mongodb.net/"


# environmental variables
env = Config()

# FastAPI configurations
fastapi_config: dict[str, Any] = {
    "title": "API",
}

mongo_url = "mongodb+srv://x6txy:baha2710@cluster0.7ohnbpk.mongodb.net/"
if env.MONGO_URL:
    mongo_url = env.MONGO_URL

# MongoDB connection
client = MongoClient(mongo_url)

# MongoDB database
database = client[env.MONGODATABASE]
