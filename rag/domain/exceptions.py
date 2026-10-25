from typing import Self


class DomainError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(DomainError):
    pass


class FileTooLargeError(DomainError):
    @classmethod
    def for_limit(cls, max_bytes: int) -> Self:
        return cls(f"Файл больше допустимых {_format_size(max_bytes)}")


def _format_size(size: int) -> str:
    megabyte = 1024 * 1024
    if size >= megabyte:
        return f"{size / megabyte:.1f}".removesuffix(".0") + " МБ"
    if size >= 1024:  # noqa: PLR2004
        return f"{size / 1024:.1f}".removesuffix(".0") + " КБ"
    return f"{size} байт"


class EmptyDocumentError(DomainError):
    pass


class TooManyPagesError(DomainError):
    pass


class DocumentNotFoundError(DomainError):
    pass


class TaskNotFoundError(DomainError):
    pass


class ExternalServiceUnavailableError(DomainError):
    pass


class InvalidModelResponseError(DomainError):
    pass
