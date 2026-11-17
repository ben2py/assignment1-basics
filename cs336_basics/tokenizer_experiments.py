from .Tokenizer import Tokenizer
import pickle
from pathlib import Path
import time
from itertools import islice
import numpy as np
import os

from concurrent.futures import ProcessPoolExecutor

def read_document(f,seperator="<|endoftext|>"):
    buffer=""
    for chunk in iter(lambda:f.read(1024*64),""):
        buffer += chunk
        while seperator in buffer:
            pos = buffer.index(seperator) + len(seperator)
            yield buffer[:pos]
            buffer = buffer[pos:]
    if buffer:
        yield buffer


def encode_afile(task):
    id,file = task
    pid = os.getpid()
    print(f"pid {pid}, task: {id}, processing file {file}")
    # t = Tokenizer.from_files("Output/openwebtext/vocab.pkl","Output/openwebtext/merges.pkl",special_tokens=["<|endoftext|>"])
    with open(file,"r",encoding="utf-8") as f:
        docs = read_document(f)
        tokens = _tokenizer.encode_iterable(docs)
        Path("Output/tinystory").mkdir(exist_ok=True,parents=True)
        start = time.perf_counter()

        output_dir = Path("Output/tinystory/tokenized")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{file.stem}.bin"

        with open(output_file,"wb") as out:
            while True:
                batch = list(islice(tokens,100000))
                # print(any([i == 256 for i in batch]))
                if not batch: break
                arr = np.asarray(batch,dtype=np.uint16)
                
                arr.tofile(out)
        elapsed = time.perf_counter() - start
        print(f"pid {pid},task: {id} ,complete file {file},takes {elapsed/60:.4f}min")

_tokenizer = None
def init_worker():
    global _tokenizer
    _tokenizer = Tokenizer.from_files("Output/tinystory/vocab.pkl","Output/tinystory/merges.pkl",special_tokens=["<|endoftext|>"])
if __name__ in "__main__":
    data_root = Path("data/tinystories")
    max_worker = 2
    task = [
        (id,file_name)
        for id,file_name in enumerate(sorted(data_root.glob("*.txt")))
    ] 
    
    
    with ProcessPoolExecutor(max_workers = max_worker,initializer=init_worker) as executor:
        for res in executor.map(encode_afile,task):
            pass
