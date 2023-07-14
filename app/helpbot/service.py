from .repository.repository import HelpBotRepository


class Service:
    def __init__(
        self,
        repository: HelpBotRepository,
    ):
        self.repository = repository


def get_service():
    repository = HelpBotRepository()
    return Service(repository)
