# import pyarrow.parquet as pq


# table = pq.read_table(path)

# print(table.schema)
# print(table.num_rows)
# print(table.column_names)

# import pyarrow.parquet as pq

# path = r"D:\Projects\assignment1-basics\data\openwebtext\train-00000-of-00080.parquet"
# table = pq.read_table(path)

# texts = table["text"].to_pylist()

import pyarrow.parquet as pq
from pathlib import Path

src_dir = Path(
    r"D:\Projects\assignment1-basics\data\openwebtext"
)

out_dir = Path(
    r"D:\Projects\assignment1-basics\data\openwebtext_txt"
)

out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(src_dir.glob("*.parquet"))

for file in files:

    print(f"Processing {file.name}")

    parquet_file = pq.ParquetFile(file)

    output_path = out_dir / (file.stem + ".txt")
    temp_path = output_path.with_suffix(".txt.tmp")

    # 每次只读取 1024 条记录
    with open(temp_path, "w", encoding="utf-8") as f:

        for batch in parquet_file.iter_batches(
            batch_size=1024,
            columns=["text"]
        ):

            texts = batch.column("text").to_pylist()

            for text in texts:

                if text is not None:
                    f.write(text)
                    f.write("\n\n")

    # 转换完成后再正式命名
    temp_path.replace(output_path)

    print(f"Saved: {output_path}")