import os
import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession

os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent
SOURCE_WAREHOUSE_DIR = BASE_DIR / "spark-warehouse"
OUTPUT_WAREHOUSE_DIR = BASE_DIR


def path_uri(path):
    return str(path.resolve()).replace("\\", "/").replace("'", "\\'")


def create_spark():
    OUTPUT_WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

    return (
        SparkSession.builder.appName("Bosch Phase 2 - Stable Parquet")
        .master("local[*]")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.sql.warehouse.dir", str(OUTPUT_WAREHOUSE_DIR))
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.network.timeout", "3600s")
        .config("spark.rpc.askTimeout", "600s")
        .config("spark.rpc.lookupTimeout", "600s")
        .config("spark.sql.shuffle.partitions", "12")
        .config("spark.default.parallelism", "12")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.inMemoryColumnarStorage.batchSize", "2000")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2000")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.parquet.columnarReaderBatchSize", "2048")
        .config("spark.sql.files.ignoreCorruptFiles", "true")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )


def source_path(source_name):
    if source_name == "categorical":
        runs_dir = SOURCE_WAREHOUSE_DIR / "bosch.db" / "categorical_runs"
        if runs_dir.exists():
            complete_runs = [
                path
                for path in runs_dir.iterdir()
                if path.is_dir() and (path / "_SUCCESS").exists()
            ]
            if complete_runs:
                return max(complete_runs, key=lambda path: path.stat().st_mtime)

    return SOURCE_WAREHOUSE_DIR / "bosch.db" / source_name


def output_path(delta_name):
    return OUTPUT_WAREHOUSE_DIR / delta_name


def parquet_files(table_path):
    return sorted(table_path.glob("part-*.parquet"))


def create_delta_table_optimized(spark, source_name, delta_name):
    print("\n" + "-" * 80)
    print(f"Processing: {source_name} -> {delta_name}")
    print("-" * 80)

    src = source_path(source_name)
    out = output_path(delta_name)

    if not src.exists():
        raise FileNotFoundError(f"Source parquet folder not found: {src}")
    output_already_written = out.exists() and (out / "_SUCCESS").exists()
    if out.exists() and not output_already_written:
        raise FileExistsError(
            f"Output path exists but is incomplete. Move or delete it before rerun: {out}"
        )

    print(f"[1] Reading source parquet: {src}", flush=True)
    df = spark.read.option("mergeSchema", "false").parquet(str(src))

    print("[2] Counting rows...", flush=True)
    start = time.time()
    row_count = df.count()
    print(f"    OK Source rows: {row_count:,} ({time.time() - start:.1f}s)", flush=True)

    num_partitions = max(8, min(24, (row_count // 100000) + 1))
    print(f"[3] Repartitioning to {num_partitions} partitions...", flush=True)
    df_optimized = df.repartition(num_partitions)

    if output_already_written:
        print(f"[4] Output already exists with _SUCCESS, reusing: {out}", flush=True)
    else:
        print(f"[4] Writing parquet: {out}", flush=True)
        start = time.time()
        (
            df_optimized.write.format("parquet")
            .mode("errorifexists")
            .option("maxRecordsPerFile", 30000)
            .option("compression", "snappy")
            .save(str(out))
        )
        print(f"    OK Write completed ({time.time() - start:.1f}s)", flush=True)

    print("[5] Verifying parquet output...", flush=True)
    files = parquet_files(out)
    total_bytes = sum(file.stat().st_size for file in files)
    if not files or total_bytes <= 0:
        raise RuntimeError(f"No parquet data files were written to {out}")
    print(f"    OK Files: {len(files):,}, size: {total_bytes:,} bytes", flush=True)

    print(f"\nOK {delta_name.upper()} - SUCCESS", flush=True)
    return out


if __name__ == "__main__":
    spark = None
    results = {}
    failures = {}

    try:
        print("=" * 80)
        print("PHASE 2 - CREATE PARQUET TABLES")
        print("=" * 80)
        print(f"Source warehouse: {SOURCE_WAREHOUSE_DIR}")
        print(f"Output directory: {OUTPUT_WAREHOUSE_DIR}")
        print("=" * 80)

        spark = create_spark()

        tables = [
            ("numeric", "delta_numeric"),
            ("date", "delta_date"),
            ("categorical", "delta_categorical"),
        ]

        for source_name, delta_name in tables:
            try:
                results[delta_name] = create_delta_table_optimized(
                    spark, source_name, delta_name
                )
            except Exception as exc:
                failures[delta_name] = exc
                print(f"\nERROR {delta_name.upper()} - FAILED: {exc}", flush=True)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for _, delta_name in tables:
            if delta_name in results:
                print(f"  {delta_name}: OK -> {results[delta_name]}")
            else:
                print(f"  {delta_name}: FAILED -> {failures.get(delta_name)}")

        if failures:
            raise RuntimeError(
                "Phase 2 failed for: " + ", ".join(sorted(failures.keys()))
            )

        print("\nPHASE 2 COMPLETED SUCCESSFULLY")
        print("=" * 80)

    finally:
        if spark is not None:
            spark.stop()
        print("\nDone!")
