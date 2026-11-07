import pyarrow.parquet as pq 
from pathlib import Path

root = Path(r"/Users/peiyanbai/Projects/stanford_cs_336/assignment1-basics/data/openwebtext")
# root = Path(r"D:\Projects\assignment1-basics\data\openwebtext")


for file in root.glob("*.parquet"):

    
    
    pf = pq.ParquetFile(root / file)
    # batch = next(pf.iter_batches(batch_size = 1000)) # a raw group

    # texts = batch.column("text").to_pylist()

    # print(texts[0])

    src = root / file
    dst_dir = Path(r"data/openwebtext_txt")
    dst = dst_dir /( file.stem+".txt")

    with open(dst,"w",encoding="utf-8") as f:
        for batch in pf.iter_batches(batch_size = 1024,columns=["text"]):
            texts = batch.column("text").to_pylist()
            for text in texts:
                
                f.write(text)
                f.write("<|endoftext|>")
                # text = batch.column(0)[0].as_py()
                # print(repr(text[:500]))
                # break
            # break