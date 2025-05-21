[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/aaronzo/pyspark-datasets/actions/workflows/ci.yml/badge.svg)](https://github.com/aaronzo/pyspark-datasets/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/aaronzo/pyspark-datasets/graph/badge.svg?token=W95YNE9KX2)](https://codecov.io/gh/aaronzo/pyspark-datasets)
-------------------------

# 📚 pyspark-datasets 📐

`pyspark-datasets` is a Python package for typed dataframes in [PySpark](https://spark.apache.org/docs/latest/api/python/index.html).
One aim of this project is to give developers type-safety similar to [Dataset API](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/Dataset.html) in Scala/Spark.

## Installation
```bash
pip install pyspark-datasets
```
At least Python 3.10 is required, with 3.12 or above recommended.  # TODO spark version

## `TypedDataFrame`
`pyspark_datasets.TypeDataFrame` is a subclass of `pyspark.sql.DataFrame` building on [sparkdantic](https://github.com/mitchelllisle/sparkdantic) which automatically derives its schema from its members, for example:
```python
from typing import Annotated
from pyspark.sql import Column
from pyspark_datasets import TypedDataFrame

class Person(TypedDataFrame):
    name: Annotated[Column, str]
    age: Annotated[Column, int]
```

```python
>>> Person.model_spark_schema()
StructType([StructField('name', StringType(), True), StructField('age', LongType(), True)])
```
Any types supported by [sparkdantic](https://github.com/mitchelllisle/sparkdantic) can be inferred, if annotated as an `Annotated[pyspark.sql.Column, ...]` field. This includes nested dataclasses, pydantic models, lists, mappings, datetimes, union and optional types, and literals, amongst other things. **Unlike to sparkdantic classes, this class also subclasses the`DataFrame` class** and thus retains all underlying methods for transformations.

An existing 'untyped' spark dataframe may be converted to a typed dataframe via the constructor:
```python
from pyspark.sql import DataFrame

df: DataFrame = ...
people = Person(df)
```
This now verifies before a spark action is called that the dataframe conforms to the schema expected by `Person`. We can optionally skip schema verification in the constructor (`verify_schema=False`), or customize it to allow missing or extra fields in the dataframe (`allow_missing_fields=True`, `allow_extra_fields=True` respectively).

Additionally, IDEs can recognise that `TypedDataFrame` instances have attributes named after their columns and syntax highlight the ones defined as fields. For example, in VSCode, the attributes become Ctrl+Clickable to jump to the class definition. In the equivalent sparkdantic model, fields would type-check as their python types and not `Column`.


### `colname` and `LazyColumn`

`pyspark_datasets.colname` is the inverse of `pyspark.sql.functions.col`: the latter takes strings and produces pyspark columns, the former retrieves column names from pyspark columns.
This can be used to increase code maintainability, using symbols instead of strings for column names where possible in conjunction with a `TypedDataFrame` instance, without the need for a second boilerplate container for column names. Consider the following example:
```python
# TODO example.
```
If a typo is made in a column name, type-checkers are unaware, and renaming fields in `Person` does not cause a type-check failure. Now instead compare this with below, where these issues are gone:

```python
# TODO example.
```
A quirk of pyspark is that creating Column instances requires an active spark context. `TypedDataFrame` circumvents this by giving `pyspark_datasets.columns.LazyColumn` instances for column attributes instead, which don't require a spark session when retreiving their column name with `colname` but subclass and behave like `pyspark.sql.Column` instances in every other way. This can allow schema validation code to sit outside scopes with active spark contexts.

### Example Usecase: `.applyInPandas`, `.mapInPandas`, and similar pyspark functions.

# TODO

### `Col`
`Col` allows defining aliases for column fields when specifying the column annotation for `TypedDataFrame`, or to separate the annotations for `TypedDataFrame` from any others the user may be using.

```python
class Items(TypedDataFrame):
    # aliased column
    _count_: Annotated[Column, Col(int, alias="count")]

    # annotated with an alias and other metadata
    name: Annotated[Column, Col(str), "<other-annotations>", ...]

```
As in the above example, aliases are often useful to annotate columns whose name would override a `DataFrame` method (in this case, `.count()`). If this field were named simply `count`, the field would be removed when defining `Items` and a warning raised to the user.

## Development
This repo uses `uv` to build the environment. Run `uv sync` to install locally and `uv build` from source. Additional checks performed by CI:
```bash
pytest --cov=pyspark_datasets tests/ --cov-report=html  # Run unit tests
mypy pyspark_datasets                                   # Type-check code
ruff check                                              # Lint code
ruff format                                             # Autoformat code
pdoc pyspark_datasets -f --html -o ./docs               # Build API reference
```

## Contributing
All contributions are welcome, especially because data engineering is not my main specialism. This project was inspired by repeatedly debugging preventable column name and type issues in ML preprocessing pipelines. Feel free to reach out through GitHub or my LinkedIn: https://www.linkedin.com/in/aaronzolnailucas/ if you are interested.
