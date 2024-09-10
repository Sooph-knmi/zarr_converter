# TODO : implement exception classes

class ArrayNotFoundError(Exception):
    def __init__(self, message="A custom error occurred"):
        self.message = message
        super().__init__(self.message)