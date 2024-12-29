import typing
import warnings
from typing import Annotated, Any, TypeVar
from pyspark.sql import DataFrame, Column, types as t
import functools as ft
import pydantic
import sparkdantic.model
from pyspark_datasets import _schema_utils
from pyspark_datasets.column import Col, LazyColumn
from pyspark_datasets.exceptions import TypedDataFrameDefinitionError, OverrideWarning

T = TypeVar("T")


class TypedDataFrameMeta(type):
    __spark_schema__: t.StructType
    __spark_field_aliases__: dict[str, str]

    def __new__(
        mcls,
        name: str,
        bases: tuple[type[Any], ...],
        attrs: dict[str, Any],
        /,
        strict_nullability: bool = False,
        **kw: Any,
    ) -> "TypedDataFrameMeta":
        cls = super().__new__(mcls, name, bases, attrs, **kw)

        dataframe_class_members = _dataframe_members()
        type_hints = typing.get_type_hints(cls, include_extras=True)

        fields: dict[str, type[Any]] = {}
        aliases: dict[str, str] = {}
        overrides = set()
        annotation_errors: list[Exception] = []

        for attr_name, attr_type in type_hints.items():
            c = _extract_annotated_column_type(attr_name, attr_type)
            match c:
                case Exception():
                    annotation_errors.append(c)
                case Col():
                    if attr_name in dataframe_class_members:
                        overrides.add(attr_name)
                        continue
                    field_name = c.alias or attr_name
                    fields[field_name] = c.field_type
                    if field_name != attr_name:
                        aliases[attr_name] = field_name
                    _set_constant_property(mcls, attr_name, LazyColumn(field_name))

        if annotation_errors:
            raise TypedDataFrameDefinitionError(
                f"Encountered {len(annotation_errors)} annotation errors deriving spark types "
                f"for dataframe class {cls.__name__}",
                annotation_errors,
            )

        if overrides:
            warnings.warn(
                f"Fields: {overrides!r} in class {cls.__name__} would override members of "
                "`pyspark.sql.DataFrame` and are being ignored. To avoid this, use an alias with the "
                "`Col` annotation.",
                OverrideWarning,
                stacklevel=2,
            )
            cls = super().__new__(mcls, name, bases, {k: v for k, v in attrs.items() if k not in overrides}, **kw)

        cls.__spark_schema__ = _create_spark_schema(fields, strict_nullability=strict_nullability)
        cls.__spark_field_aliases__ = aliases
        return cls


def _annotated_type_and_metadata(typ: Annotated[Any, ...]) -> tuple[type[Any], tuple[Any, ...]]:
    return getattr(typ, "__origin__", type(None)), getattr(typ, "__metadata__", ())


def _extract_annotated_column_type(name: str, type_hint: type[Any]) -> "Col[Any] | None | TypeError":
    origin = typing.get_origin(type_hint)
    if origin is not Annotated:
        return None
    annotated, annotations = _annotated_type_and_metadata(type_hint)
    if not issubclass(annotated, Column):
        return None

    if len(annotations) == 1:
        c, *_ = annotations
        return c if isinstance(c, Col) else Col(c)

    column_annotations = [c for c in annotations if isinstance(c, Col)]
    match column_annotations:
        case (c,):
            return c
        case _:
            return TypeError(
                f"field {name} must have exactly one `Col` annotation (annotation found: `{type_hint}`)."
                "If only one annotation is given, it will be used as the column type."
            )


def _create_spark_schema(fields: dict[str, type[Any]], strict_nullability: bool) -> t.StructType:
    _SparkModel = pydantic.create_model("_SparkModel", **{k: (v, ...) for k, v in fields.items()})  # type: ignore[call-overload]
    schema = sparkdantic.model.create_spark_schema(_SparkModel, safe_casting=True)
    del _SparkModel
    if not strict_nullability:
        _schema_utils.to_nullable_inplace(schema)
    return schema


@ft.cache
def _dataframe_members() -> set[str]:
    return set(dir(DataFrame))


def _set_constant_property(cls: type[Any], name: str, value: Any) -> None:
    setattr(cls, name, property(ft.partial(_constant, value)))


def _constant(value: T, *_: Any) -> T:
    return value
