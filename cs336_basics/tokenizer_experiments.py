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

# data_root = Path("data")
# weight_root = Path("Output")

# with open(weight_root / "vocab.pkl","rb") as f:
#     tiny_vocab = pickle.load(f)

# with open(weight_root / "merges.pkl","rb") as f:
#     tiny_merges = pickle.load(f)

# with open(weight_root / "openwebtext/vocab.pkl","rb") as f:
#     web_vocab = pickle.load(f)

# with open(weight_root / "openwebtext/merges.pkl","rb") as f:
#     web_merges = pickle.load(f)


# t_tiny = Tokenizer(tiny_vocab,tiny_merges,special_tokens=["<|endoftext|>"])
# t_web = Tokenizer(web_vocab,web_merges,special_tokens=["<|endoftext|>"])
# # t_tiny = Tokenizer.from_files(weight_root / "vocab.pkl",weight_root / "merges.pkl",special_tokens=)

# with open(data_root/"openwebtext_txt/train-00000-of-00080.txt") as f:
#     docs = []
#     encoded_docs = []
#     encoded_tiny = []
#     encoded_web = []
#     for i in range(10):
#         doc = next(read_document(f))

#         docs.extend(doc)
#         encoded_tiny.extend(t_tiny.encode(doc))
#         encoded_web.extend(t_web.encode(doc))
#         encoded_docs.extend(doc.encode("utf-8"))
    
    
#     print(f"the compression ration of tinystory_tokenizer in OpenWebText is {len(encoded_docs)/len(encoded_tiny)}")
#     print(f"the compression ration of openwebtext_tokenizer in OpenWebText is {len(encoded_docs)/len(encoded_web)}")
#     # print(t_tiny.encode(doc))


# with open(data_root/"data/TinyStoriesV2-GPT4-train.txt") as f:
#     docs = []
#     encoded_docs = []
#     encoded_tiny = []
#     encoded_web = []
#     for i in range(10):
#         doc = next(read_document(f))

#         docs.extend(doc)
#         encoded_tiny.extend(t_tiny.encode(doc))
#         encoded_web.extend(t_web.encode(doc))
#         encoded_docs.extend(doc.encode("utf-8"))
    
    
#     print(f"the compression ration of tinystory_tokenizer in tinystory is {len(encoded_docs)/len(encoded_tiny)}")
#     print(f"the compression ration of openwebtext_tokenizer in tinystory is {len(encoded_docs)/len(encoded_web)}")
        
# #compute the throuput(in bytes/second)

# # with open(data_root/"openwebtext_txt/train-00000-of-00080.txt") as f:
# with open(data_root/"data/TinyStoriesV2-GPT4-train.txt") as f:

#     docs = read_document(f)
#     tokens = t_tiny.encode_iterable(docs)
#     Path(weight_root/"tinystory").mkdir(exist_ok=True)
#     start = time.perf_counter()
#     total_token = 0
#     with open(weight_root/"tinystory"/"train_tinystory.bin","wb") as out:
#         while True:
#             batch = list(islice(tokens,100_000))
#             if not batch:
#                 break
                
#             if min(batch) < 0 or max(batch) > 2**16 :
#                 raise RecursionError("Token id exceeds uint16 range")
            
#             total_token += len(batch)

#             arr = np.asarray(batch,dtype=np.uint16)
#             arr.tofile(out)

#             elapsed = time.perf_counter() - start
#             print(f"total_token: {total_token}, time consumed is {elapsed}, token/sec is {total_token/elapsed}",flush=True)
        
#     info = t_tiny._encode_pretoken.cache_info()
#     print(f"hit rate: {info.hits()/(info.hits()+info.passes())}")

def encode_afile(task):
    id,file = task
    pid = os.getpid()
    print(f"pid {pid}, task: {id}, processing file {file}")
    t = Tokenizer.from_files("Output/vocab.pkl","Output/merges.pkl",special_tokens=["<|endoftext|>"])
    with open(file,"r") as f:
        docs = read_document(f)
        tokens = t.encode_iterable(docs)
        Path("Output/openwebtext").mkdir(exist_ok=True,parents=True)
        start = time.perf_counter()
        with open("Output/openwebtext/train.bin","wb") as out:
            while True:
                batch = list(islice(tokens,100000))
                if not batch: break
                
                arr = np.asarray(batch,dtype=np.uint16)
                arr.tofile(out)
        elapsed = time.perf_counter() - start
        print(f"pid {pid},task: {id} ,complete file {file},takes {elapsed/60:.4f}min")
if __name__ in "__main__":
    data_root = Path("data/openwebtext_txt")
    max_worker = 10
    task = {
        (id,file_name)
        for id,file_name in enumerate(list(data_root.glob("*.txt")))
    } 
    print(task)
    with ProcessPoolExecutor(max_workers = max_worker) as executor:
        for res in executor.map(encode_afile,task):
            pass
