import pyspark.sql.types as t
from pyspark_datasets import _schema_utils


def test_to_nullable_inplace_struct():
    a = t.StructType(
        [t.StructField("a1", t.IntegerType(), nullable=True), t.StructField("a2", t.StringType(), nullable=False)]
    )
    _schema_utils.to_nullable_inplace(a)
    assert all(f.nullable for f in a)


def test_to_nullable_inplace_array():
    a = t.ArrayType(t.IntegerType(), containsNull=True)
    b = t.ArrayType(t.DoubleType(), containsNull=False)
    _schema_utils.to_nullable_inplace(a)
    _schema_utils.to_nullable_inplace(b)
    assert a.containsNull
    assert b.containsNull


def test_to_nullable_inplace_map():
    a = t.MapType(t.IntegerType(), t.StringType(), valueContainsNull=True)
    b = t.MapType(t.FloatType(), t.BooleanType(), valueContainsNull=False)
    _schema_utils.to_nullable_inplace(a)
    _schema_utils.to_nullable_inplace(b)
    assert a.valueContainsNull
    assert b.valueContainsNull


def test_to_nullable_inplace_noop():
    a = t.StringType()
    assert _schema_utils.to_nullable_inplace(a) is None


def assert_nullable_inplace_nested():
    a = t.StructType(
        [
            t.StructField("a1", t.StructType([t.StructField("a1.b", t.StringType(), nullable=True)]), nullable=False),
            t.StructField("a2", t.ArrayType(t.IntegerType(), containsNull=False), nullable=False),
            t.StructField("a3", t.MapType(t.IntegerType(), t.StringType(), valueContainsNull=False), nullable=False),
        ]
    )
    expected = t.StructType(
        [
            t.StructField("a1", t.StructType([t.StructField("a1.b", t.StringType(), nullable=True)]), nullable=True),
            t.StructField("a2", t.ArrayType(t.IntegerType(), containsNull=True), nullable=True),
            t.StructField("a3", t.MapType(t.IntegerType(), t.StringType(), valueContainsNull=True), nullable=True),
        ]
    )
    assert a != expected
    _schema_utils.to_nullable_inplace(a)
    assert a == expected


def test_schema_diff_nullable():
    a = _schema_utils.schema_diff(
        t.StructType([t.StructField("x", t.IntegerType(), nullable=True)]),
        t.StructType([t.StructField("x", t.IntegerType(), nullable=True)]),
        ignore_nullable=False,
    )
    assert len(a) == 0

    b = _schema_utils.schema_diff(
        t.StructType([t.StructField("x", t.IntegerType(), nullable=True)]),
        t.StructType([t.StructField("x", t.IntegerType(), nullable=False)]),
        ignore_nullable=False,
    )
    assert len(b) == 1

    c = _schema_utils.schema_diff(
        t.StructType([t.StructField("x", t.IntegerType(), nullable=True)]),
        t.StructType([t.StructField("x", t.IntegerType(), nullable=False)]),
        ignore_nullable=True,
    )
    assert len(c) == 0


def test_schema_diff_metadata():
    a = _schema_utils.schema_diff(
        t.StructType([t.StructField("x", t.IntegerType(), metadata={"y": 1})]),
        t.StructType([t.StructField("x", t.IntegerType(), metadata={"y": 1})]),
        ignore_metadata=False,
    )
    assert len(a) == 0

    b = _schema_utils.schema_diff(
        t.StructType([t.StructField("x", t.IntegerType(), metadata={"y": 1})]),
        t.StructType([t.StructField("x", t.IntegerType(), metadata=None)]),
        ignore_metadata=False,
    )
    assert len(b) == 1

    c = _schema_utils.schema_diff(
        t.StructType([t.StructField("x", t.IntegerType(), metadata={"y": 1})]),
        t.StructType([t.StructField("x", t.IntegerType(), metadata=None)]),
        ignore_metadata=True,
    )
    assert len(c) == 0


def test_schema_diff_extra_fields_in_source():
    a = _schema_utils.schema_diff(
        t.StructType(
            [t.StructField("x", t.IntegerType(), nullable=True), t.StructField("y", t.IntegerType(), nullable=True)]
        ),
        t.StructType([t.StructField("x", t.IntegerType(), nullable=True)]),
    )
    assert len(a) == 1

    b = _schema_utils.schema_diff(
        t.StructType(
            [t.StructField("x", t.IntegerType(), nullable=True), t.StructField("y", t.IntegerType(), nullable=True)]
        ),
        t.StructType([t.StructField("x", t.IntegerType(), nullable=True)]),
        allow_extra_fields_in_source=True,
    )
    assert len(b) == 0


def test_schema_diff_extra_fields_in_target():
    a = _schema_utils.schema_diff(
        t.StructType([t.StructField("x", t.IntegerType(), nullable=True)]),
        t.StructType(
            [t.StructField("x", t.IntegerType(), nullable=True), t.StructField("y", t.IntegerType(), nullable=True)]
        ),
    )
    assert len(a) == 1

    b = _schema_utils.schema_diff(
        t.StructType([t.StructField("x", t.IntegerType(), nullable=True)]),
        t.StructType(
            [t.StructField("x", t.IntegerType(), nullable=True), t.StructField("y", t.IntegerType(), nullable=True)]
        ),
        allow_extra_fields_in_target=True,
    )
    assert len(b) == 0


def test_schema_diff_nested():
    a = _schema_utils.schema_diff(
        t.StructType([t.StructField("x", t.StructType([t.StructField("y", t.IntegerType())]))]),
        t.StructType([t.StructField("x", t.StructType([t.StructField("y", t.IntegerType())]))]),
    )
    assert len(a) == 0

    b = _schema_utils.schema_diff(
        t.StructType([t.StructField("x", t.StructType([t.StructField("y", t.IntegerType())]))]),
        t.StructType([t.StructField("x", t.StructType([t.StructField("y", t.StringType())]))]),
    )
    assert len(b) == 1
