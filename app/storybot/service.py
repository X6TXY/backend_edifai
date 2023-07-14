from .repository.repository import StoryBotRepository


class Service:
    def __init__(
        self,
        repository: StoryBotRepository,
    ):
        self.repository = repository


def get_service():
    repository = StoryBotRepository()
    return Service(repository)
