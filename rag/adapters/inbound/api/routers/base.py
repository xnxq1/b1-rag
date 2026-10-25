from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar, cast

from fastapi import APIRouter


class BaseRouter(ABC):
    prefix: str = ""
    tags: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self.router = APIRouter(prefix=self.prefix, tags=cast("list[str | Enum]", self.tags))
        self._setup_router_endpoints(self.router)

    @abstractmethod
    def _setup_router_endpoints(self, router: APIRouter) -> None: ...
