
class BrowserInitException(Exception):
    def __init__(
        self, message: str = 'Failed to initialize browser environment'
    ) -> None:
        super().__init__(message)
