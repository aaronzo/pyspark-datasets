"""Column annotation and utilities for `TypedDataFrame` instances."""

from typing import Any, Protocol, TypeVar, Generic
from pyspark.sql import Column
import functools as ft
import pyspark.sql.functions as F
import typing
import dataclasses as dc


_T = TypeVar("_T", bound=type[Any])

def colname(column: Column) -> str:
    """Get the string name of a spark column, even if it is a lazy column.

    This is particularly useful for referencing fields in TypedDataFrames symbolically
    for type-checking and future-proofing refactoring.

    If the column is a lazy column, an active spark session is not required to obtain
    its name.

    Args:
        column (Column): The column to get the name of.

    Returns:
        str: The name of the column.
    """
    if isinstance(column, LazyColumn):
        return column.__column_name__
    return str(typing.cast(_JavaColumn, column._jc).toString())


@dc.dataclass
class Col(Generic[_T]):
    """Column annotation for use in `TypedDataFrame` subclasses.

    Allows defining aliases for column fields when specifying a column annotation
    or to separate the annotations for `TypedDataFrame` from any others the user may be using.

    Attributes:
        field_type (type[Any]): The type of the column.
        alias (str | None, optional): The alias of the column. Defaults to None.
    """

    field_type: _T
    alias: str | None = dc.field(default=None, kw_only=True)


class LazyColumn(Column):
    """A lazy column, whose name can be referenced without an active spark session.

    Args:
        column_name (str): The name of the column.
    """

    def __init__(self, column_name: str) -> None:
        self.__column_name__ = column_name

    @ft.cached_property
    def _get(self) -> Column:
        return F.col(self.__column_name__)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get, name)


class _JavaColumn(Protocol):
    def toString(self) -> str: ...
