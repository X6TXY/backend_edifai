from app.config import database

from .repository.repository import WritingTask2


class Service:
    def __init__(
        self,
        repository: WritingTask2,
    ):
        self.repository = repository


def get_service():
    repository = WritingTask2(database)
    return Service(repository)
