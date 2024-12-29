"""Module for the `TypedDataFrame` class."""

from typing import overload, Literal
from pyspark.sql import DataFrame, Column, types as t
from pyspark_datasets._dataframe import TypedDataFrameMeta
from pyspark_datasets._schema_utils import schema_diff
from pyspark_datasets.exceptions import SchemaVerificationError


class TypedDataFrame(DataFrame, metaclass=TypedDataFrameMeta):
    """DataFrame with type information for automatic schema derivation and verification.

    Args:
        df (DataFrame): The plain spark DataFrame to wrap.
        verify_schema (bool): Whether to verify the schema of the DataFrame against the schema
            derived from the type hints of the DataFrame class.
        **verify_schema_kw: Additional keyword arguments to pass to the `verify_schema` method.
            Only used if `verify_schema` is True.
    """

    __spark_schema__: t.StructType
    __spark_field_aliases__: dict[str, str]

    @overload
    def __init__(self, df: DataFrame, verify_schema: Literal[True] = True, **verify_schema_kw: bool) -> None: ...

    @overload
    def __init__(self, df: DataFrame, verify_schema: Literal[False] = False) -> None: ...

    def __init__(
        self,
        df: DataFrame,
        verify_schema: bool = True,
        **verify_schema_kw: bool,
    ) -> None:
        super().__init__(df._jdf, df.sparkSession)
        if verify_schema:
            self.verify_schema(**verify_schema_kw)

    def verify_schema(
        self,
        *,
        allow_extra_fields: bool = False,
        allow_missing_fields: bool = False,
        ignore_nullable: bool = True,
        ignore_metadata: bool = True,
    ) -> None:
        """Verify the schema of the DataFrame against the schema derived from the type hints of the DataFrame class.

        Args:
            allow_extra_fields (bool): Whether to allow extra fields in the DataFrame schema.
            allow_missing_fields (bool): Whether to allow missing fields in the DataFrame schema.
            ignore_nullable (bool): Whether to ignore nullable fields in the DataFrame schema.
            ignore_metadata (bool): Whether to ignore metadata in the DataFrame schema.

        Raises:
            SchemaVerificationError: If the schema of the DataFrame does not match the schema derived
                from the type hints of the DataFrame class. The error message will contain a list of
                individual errors indicating the differences between the schemas.
        """
        errors = schema_diff(
            self.schema,
            self.model_spark_schema(),
            allow_extra_fields_in_source=allow_extra_fields,
            allow_extra_fields_in_target=allow_missing_fields,
            ignore_nullable=ignore_nullable,
            ignore_metadata=ignore_metadata,
        )
        if errors:
            raise SchemaVerificationError(
                "Schema verification failed with the following errors:",
                errors,
            )

    @classmethod
    def model_spark_schema(cls) -> t.StructType:
        """Returns the PySpark schema derived from the type hints of the DataFrame class.

        Returns:
            pyspark.sql.types.StructType: The PySpark schema derived from the type hints of the DataFrame class.
        """
        return cls.__spark_schema__

    def __getattr__(self, name: str) -> Column:
        if (alias := self.__spark_field_aliases__.get(name)) is not None:
            return super().__getattr__(alias)
        return super().__getattr__(name)
