from pyspark_datasets.column import Col, colname, LazyColumn
import pyspark.sql.functions as F
from pyspark.sql import Column
import pytest


@pytest.mark.parametrize("column_name", ["", "x", "y"])
def test_colname_on_lazy_column_no_spark(column_name: str):
    assert colname(LazyColumn(column_name)) == column_name


@pytest.mark.usefixtures("sc")
@pytest.mark.parametrize("column_name", ["", "x", "y"])
def test_colname_on_spark_column(column_name: str):
    assert colname(F.col(column_name)) == column_name


def test_col_init():
    a = Col(str)
    assert a.alias is None
    assert a.field_type is str

    b = Col(str, alias="b")
    assert b.alias == "b"
    assert b.field_type is str

    c = Col(int, alias=None)
    assert c.alias is None
    assert c.field_type is int


def test_lazy_column_getattr():
    assert isinstance(~LazyColumn("x"), Column)
