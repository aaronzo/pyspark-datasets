"""Exception classes for _pyspark-datasets_."""

import sys

if sys.version_info < (3, 11):  # pragma: no cover
    from exceptiongroup import ExceptionGroup


class SchemaVerificationError(ExceptionGroup):
    """Grouped schema verification errors."""

    ...


class TypedDataFrameDefinitionError(ExceptionGroup):
    """Grouped errors deriving spark types for a `TypedDataFrame` class."""

    ...


class OverrideWarning(UserWarning):
    """Warning raised when overriding fields in a parent class."""

    ...
