"""A package bringing type-based schema derivation and verification to PySpark DataFrames, building on SparkDantic."""

from pyspark_datasets.column import Col, colname
from pyspark_datasets.dataframe import TypedDataFrame

__all__ = [
    "colname",
    "Col",
    "TypedDataFrame",
]

__version__ = "0.1.0"
