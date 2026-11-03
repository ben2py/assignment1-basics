import pyarrow.parquet as pq 
from pathlib import Path

root = Path(r"/Users/peiyanbai/Projects/stanford_cs_336/assignment1-basics/data/openwebtext_text")
file_name = "train-00000-of-00080.parquet"


pf = pq.ParquetFile(root / file_name)
# batch = next(pf.iter_batches(batch_size = 1000)) # a raw group

# texts = batch.column("text").to_pylist()

# print(texts[0])

src = file_name
dst = (root/file_name).with_suffix(".txt")

with open(dst,"w",encoding="utf-8") as f:
    for batch in pf.iter_batches(batch_size = 1024,columns=["text"]):
        texts = batch.column("text").to_pylist()
        for text in texts:
            f.write(text)
            f.write("<|endoftext|>")