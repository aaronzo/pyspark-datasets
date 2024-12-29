import pytest
import typing
from typing import Iterator
from pyspark.sql import SparkSession
from pyspark import SparkContext


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    spark = (
        typing.cast(SparkSession.Builder, SparkSession.builder)
        .appName("pyspark-datasets-tests")
        .config("spark.default.parallelism", "2")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def sc() -> Iterator[SparkContext]:
    sc = SparkContext.getOrCreate()
    yield sc
    sc.stop()
