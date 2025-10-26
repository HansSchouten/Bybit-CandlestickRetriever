from main import DEBUG_MODE
import pandas as pd
import os


def debug(msg):
    if DEBUG_MODE:
        print(msg)


def clean(df):
    """clean a raw dataframe"""
    df = remove_no_volume_rows(df)

    df = df.drop_duplicates(subset=['datetime'], keep='last')

    df.sort_values(by=['datetime'], ascending=False)
    df.reset_index(drop=True, inplace=True)

    assert_integrity(df)

    return df


def remove_no_volume_rows(df):
    """Return rows with numeric volume > 0."""
    vol = pd.to_numeric(df['volume'], errors='coerce')

    valid_mask = vol > 0
    df_cleaned = df.loc[valid_mask].copy()

    df_cleaned['volume'] = vol.loc[valid_mask]

    return df_cleaned


def assert_integrity(df):
    """make sure no rows have empty cells or duplicate timestamps exist"""

    assert df.isna().all(axis=1).any() == False
    assert df['datetime'].duplicated().any() == False


def write_raw_to_parquet(df, full_path):
    """takes raw df and writes a parquet to disk"""

    df_copy = set_dtypes_compressed(df)
    debug("Before writing:")
    debug(df_copy)
    df_copy.to_parquet(
        full_path,
        engine="pyarrow",
        compression="zstd",
        compression_level=7,
        index=False,
        use_dictionary=False,
        coerce_timestamps="ms",
        allow_truncated_timestamps=True,
    )


def set_dtypes_compressed(df):
    """Create a `DatetimeIndex` and convert all critical columns in pd.df to a dtype with low memory profile."""

    df_copy = df.copy()
    df_copy = df_copy.astype(dtype={
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'float32',
    })

    # ensure clean ms precision UTC timestamps
    df_copy['datetime'] = pd.to_datetime(df_copy['datetime'], utc=True).dt.floor('ms')

    return df_copy[['datetime', 'open', 'high', 'low', 'close', 'volume']]


def groom_all(dirname='compressed'):
    """go through `dirname` and perform a quick clean on all `.parquet` files"""

    for filename in os.listdir(dirname):
        if filename.endswith('.parquet'):
            full_path = os.path.join(dirname, filename)
            df = pd.read_parquet(full_path)
            debug(df)
            df_clean = clean(df)
            debug(df_clean)
            write_raw_to_parquet(df_clean, full_path)