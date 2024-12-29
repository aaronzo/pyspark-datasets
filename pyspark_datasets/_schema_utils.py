import pyspark.sql.types as t


def to_nullable_inplace(dtype: t.DataType) -> None:
    match dtype:
        case t.StructType():
            for field in dtype:
                to_nullable_inplace(field.dataType)
                field.nullable = True
        case t.ArrayType():
            to_nullable_inplace(dtype.elementType)
            dtype.containsNull = True
        case t.MapType():
            to_nullable_inplace(dtype.keyType)
            to_nullable_inplace(dtype.valueType)
            dtype.valueContainsNull = True


def schema_diff(
    source_schema: t.StructType,
    target_schema: t.StructType,
    *,
    allow_extra_fields_in_source: bool = False,
    allow_extra_fields_in_target: bool = False,
    ignore_nullable: bool = True,
    ignore_metadata: bool = True,
) -> list[ValueError]:
    return _schema_diff(
        source_schema,
        target_schema,
        allow_extra_fields_in_source=allow_extra_fields_in_source,
        allow_extra_fields_in_target=allow_extra_fields_in_target,
        ignore_nullable=ignore_nullable,
        ignore_metadata=ignore_metadata,
    )


def _schema_diff(
    source_schema: t.StructType,
    target_schema: t.StructType,
    root_name: str = "",
    *,
    allow_extra_fields_in_source: bool,
    allow_extra_fields_in_target: bool,
    ignore_nullable: bool,
    ignore_metadata: bool,
) -> list[ValueError]:
    diffs: list[ValueError] = []
    source_field_names = set(source_schema.fieldNames())
    target_field_names = set(target_schema.fieldNames())

    if (extra := source_field_names - target_field_names) and not allow_extra_fields_in_source:
        diffs.append(ValueError("Extra fields: " + ",".join(f"{root_name}{f}" for f in extra)))
    if (missing := target_field_names - source_field_names) and not allow_extra_fields_in_target:
        diffs.append(ValueError("Missing fields: " + ",".join(f"{root_name}{f}" for f in missing)))

    for field in source_field_names & target_field_names:
        a, b = source_schema[field], target_schema[field]

        if isinstance(a.dataType, t.StructType) and isinstance(b.dataType, t.StructType):
            diffs.extend(
                _schema_diff(
                    a.dataType,
                    b.dataType,
                    f"{root_name}{field}.",
                    allow_extra_fields_in_source=allow_extra_fields_in_source,
                    allow_extra_fields_in_target=allow_extra_fields_in_target,
                    ignore_nullable=ignore_nullable,
                    ignore_metadata=ignore_metadata,
                )
            )

        elif a.dataType != b.dataType:
            diffs.append(ValueError(f"Field {root_name}{field} has different type"))

        if not ignore_nullable and a.nullable != b.nullable:
            diffs.append(ValueError(f"Field {root_name}{field} has different nullable value"))

        if not ignore_metadata and a.metadata != b.metadata:
            diffs.append(ValueError(f"Field {root_name}{field} has different metadata"))

    return diffs
