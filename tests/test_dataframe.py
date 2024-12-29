from pyspark_datasets import TypedDataFrame, colname, Col
from pyspark.sql import SparkSession, Column, DataFrame, Row
from typing import Annotated, Protocol
import pytest
import pyspark.sql.functions as F
from pyspark_datasets import exceptions
import pyspark.sql.types as t
from sparkdantic.model import SparkModel


class Person(Protocol):
    name: Column
    age: Column


class Group(Protocol):
    people: list[Person]
    _count_: Col[int]
    is_root: bool


@pytest.fixture(scope="module")
def person_dataframe_cls() -> type[Person]:
    class Person(TypedDataFrame):
        name: Annotated[Column, str]
        age: Annotated[Column, int]

    return Person


@pytest.fixture(scope="module")
def group_dataframe_cls() -> type[Group]:
    class Person(SparkModel):
        name: str
        age: int

    class Group(TypedDataFrame):
        people: Annotated[Column, list[Person]]
        _count_: Annotated[Column, Col(int, alias="count")]
        is_root: bool = True

    return Group


@pytest.fixture(scope="module")
def person_df(person_dataframe_cls: type[Person | TypedDataFrame], spark: SparkSession) -> Person | TypedDataFrame:
    return person_dataframe_cls(
        spark.createDataFrame([("a", 1), ("b", 2)], schema=["name", "age"]),
        verify_schema=False,
    )


@pytest.fixture(scope="module")
def group_df(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [
            ([Row(name="a", age=1), Row(name="b", age=2)], 2),
            ([Row(name="c", age=3), Row(name="d", age=4)], 4),
        ],
        schema=["people", "count"],
    )


def test_empty_typed_dataframe():
    class Empty(TypedDataFrame): ...

    assert not Empty.model_spark_schema().fields


def test_colname_on_typed_dataframe_class(person_dataframe_cls: type[Person | TypedDataFrame]):
    assert colname(person_dataframe_cls.name) == "name"
    assert colname(person_dataframe_cls.age) == "age"


def test_colname_on_typed_dataframe_instance(person_df: Person | TypedDataFrame):
    assert colname(person_df.name) == "name"
    assert colname(person_df.age) == "age"


def test_schema_correct(person_df: Person | TypedDataFrame):
    person_df.verify_schema()
    person_df.__class__(person_df, verify_schema=True)


def test_schema_incorrect(person_df: Person | TypedDataFrame):
    with pytest.raises(exceptions.SchemaVerificationError, match="Schema verification failed") as e:
        person_df.__class__(
            person_df.withColumn("name", F.lit(False)).withColumn("extra", F.lit(1)),
            verify_schema=True,
        )

    schema_verification_errors = e.value.exceptions
    assert len(schema_verification_errors) == 2
    assert all(isinstance(e, ValueError) for e in schema_verification_errors)


def test_typed_dataframe_with_aliases(person_df: Person | TypedDataFrame):
    class Person(TypedDataFrame):
        x1: Annotated[Column, Col(str, alias="name")]
        x2: Annotated[Column, Col(int, alias="age")]

    df = Person(person_df, verify_schema=True)
    isinstance(df.x1, Column)
    isinstance(df.x2, Column)


def test_typed_dataframe_with_overrides():
    with pytest.warns(exceptions.OverrideWarning, match="Fields: {'count'} in class _ would override"):

        class _(TypedDataFrame):
            count: Annotated[Column, str]


def test_typed_dataframe_ignored_attributes():
    class _(TypedDataFrame):
        a: str | int
        b: Annotated[float, "not a column"]
        c: Annotated[Column, Col(str), "other metadata"]


def test_typed_dataframe_error_on_multiple_annotations():
    with pytest.raises(
        exceptions.TypedDataFrameDefinitionError, match="annotation errors deriving spark types for dataframe class"
    ) as e:

        class _(TypedDataFrame):
            a: Annotated[Column, Col(str), Col(int)]
            b: Annotated[Column, "a", "b"]

    annotation_errors = e.value.exceptions
    assert len(annotation_errors) == 2
    assert all(isinstance(e, TypeError) for e in annotation_errors)


def test_typed_dataframe_extra_fields(group_dataframe_cls: type[Group | TypedDataFrame], group_df: DataFrame):
    group_df = group_df.withColumn("extra", F.lit(1))
    with pytest.raises(exceptions.SchemaVerificationError, match="Schema verification failed") as e:
        group_dataframe_cls(group_df, verify_schema=True)

    assert len(e.value.exceptions) == 1
    group_dataframe_cls(group_df, verify_schema=True, allow_extra_fields=True)


def test_typed_dataframe_missing_fields(group_dataframe_cls: type[Group | TypedDataFrame], group_df: DataFrame):
    group_df = group_df.drop("count")

    with pytest.raises(exceptions.SchemaVerificationError, match="Schema verification failed") as e:
        group_dataframe_cls(group_df, verify_schema=True)

    assert len(e.value.exceptions) == 1
    group_dataframe_cls(group_df, verify_schema=True, allow_missing_fields=True)


def test_typed_dataframe_nested(group_dataframe_cls: type[Group | TypedDataFrame]):
    assert group_dataframe_cls.model_spark_schema() == t.StructType(
        [
            t.StructField(
                "people",
                t.ArrayType(
                    t.StructType(
                        [
                            t.StructField("name", t.StringType()),
                            t.StructField("age", t.LongType()),
                        ]
                    )
                ),
            ),
            t.StructField("count", t.LongType()),
        ]
    )
