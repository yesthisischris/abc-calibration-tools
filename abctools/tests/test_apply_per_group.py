import polars as pl
from abctools.abc_classes import apply_per_group_preserve_key


def test_apply_per_group_multiple_columns():
    df = pl.DataFrame({"group": ["a", "a", "b"], "x": [1, 2, 3]})

    def udf(subdf: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "x_sum": [subdf["x"].sum()],
                "x_mean": [subdf["x"].mean()],
            }
        )

    result = apply_per_group_preserve_key(df, key="group", user_udf=udf)

    assert set(result.columns) == {"group", "x_sum", "x_mean"}
    assert result.filter(pl.col("group") == "a")["x_sum"].item() == 3
    assert result.filter(pl.col("group") == "b")["x_sum"].item() == 3


def test_apply_per_group_scalar():
    df = pl.DataFrame({"group": ["a", "a", "b"], "x": [1, 2, 3]})

    def udf(subdf: pl.DataFrame) -> int:
        return subdf["x"].sum()

    result = apply_per_group_preserve_key(
        df, key="group", user_udf=udf, result_column="total"
    )

    assert result.columns == ["group", "total"]
    assert result.filter(pl.col("group") == "a")["total"].item() == 3
    assert result.filter(pl.col("group") == "b")["total"].item() == 3
