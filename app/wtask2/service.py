from .repository.repository import Wtask2Repository


class Service:
    def __init__(
        self,
        repository: Wtask2Repository,
    ):
        self.repository = repository


def get_service():
    repository = Wtask2Repository()
    return Service(repository)
