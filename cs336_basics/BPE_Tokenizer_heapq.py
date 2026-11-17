from .pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter,defaultdict
from typing import BinaryIO
import pickle
from pathlib import Path

import time
from concurrent.futures import ProcessPoolExecutor
import os
import heapq
from dataclasses import dataclass
from tqdm import tqdm

@dataclass(frozen = True)
class HeapEntry:
    freq:int
    pair:tuple[bytes,bytes]
    version:int
    def __lt__(self,other):
        return (self.freq,self.pair) > (other.freq,other.pair)

def update_pair_freq(pair_freq,pair,delta):
    
    new_freq = pair_freq.get(pair,0) + delta
    if new_freq < 0:
        raise RuntimeError(f"Negative pair frequency: {pair}")
    versions[pair] +=1

    if new_freq == 0:
        pair_freq.pop(pair)
        return
    pair_freq[pair] = new_freq
    heapq.heappush(heap,HeapEntry(new_freq,pair,versions[pair]))
    
    
# def pretokenization(path:str,max_dividing:int,special_tokens:BinaryIO):
#     ct = Counter() #record how many times tuple of each word emerge {('a','b','c'):100}
#     with open(path,"rb") as f:
#         boundries = find_chunk_boundaries(f,max_dividing,special_tokens)
#         cnt = 1
#         pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# )
#         for start,end in zip(boundries[:-1],boundries[1:]):
#             f.seek(start)
#             chunk = f.read(end-start).decode("utf-8",errors = "ignore")
#             for part in re.split(r"<\|endoftext\|>",chunk):
#                 chunk_ = pattern.findall(part)
#                 # temp = [tuple(bytes([b])  for b in s.encode("utf-8")) for s in chunk_]
#                 # 将str pretoken 先统计完，完后在将键转化成为单字符元组，会更高效
#                 # s.encode("utf-8")后 b是int 类型的；bytes([])之后，b会变成 b'b'的格式，可以像字符串拼接
#                 ct.update(chunk_)

#             # temp = [tuple(bytes([b])  for b in s.encode("utf-8")) for s in chunk_]
#             # s.encode("utf-8")后 b是int 类型的；bytes([])之后，b会变成 b'b'的格式，可以像字符串拼接
#             print(f"has processed to {cnt}")
#             cnt +=1
            
#     return ct

def get_best_pair(heap,pair_freq,versions):
    while heap:
        entry = heapq.heappop(heap)

        if versions[entry.pair] != entry.version:
            continue
        if pair_freq.get(entry.pair,0) != entry.freq:
            continue
        return entry.pair
    raise RuntimeError("No valid pairs remain")


def pretokenization(task):
    chunk_id,file_path,start,end = task
    pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
    ct = Counter()
    pid = os.getpid()
    begin = time.perf_counter()
    print(
        f"[PID {pid}] 开始块 {chunk_id}，"
        f"大小 {(end - start) / 1024**2:.1f} MiB",
        flush=True,
    )
    with open(file_path,"rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8",errors = "ignore")
        for part in re.split(r"<\|endoftext\|>",chunk):
            chunk_ = pattern.findall(part)
            ct.update(chunk_)
    print(
        f"[PID {pid}] 完成块 {chunk_id}，"
        f"耗时 {time.perf_counter() - begin:.2f}s，"
        f"不同词数 {len(ct)}",
        flush=True,
    )

    return ct

if __name__ == "__main__":
    root = Path("data/tinystories")
    # root = Path(r"D:\Projects\assignment1-basics\data\openwebtext_txt")
    
    vocabulary = {
        i:bytes([i]) for i in range(256)
    }
    special_tokens = [b"<|endoftext|>"]
    special_characters = [",",".","!","@","#","$","%","^","&","*","(",")","-","+","=","/",";",":","\'","\"","\n","\t"," ","\r"]
    max_vocabsize = 10000
    num_chunks = 128
    max_worker = 8
    vocabulary[len(vocabulary)] = b"<|endoftext|>"
    ct = Counter()
    begin= time.perf_counter()
    print(list(root.glob("*.txt")))
    for file in root.glob("*.txt"):
        print(file)
        path = file
        boundries = list()
        with open(path,"rb") as f:
            boundries = find_chunk_boundaries(f,num_chunks,b"<|endoftext|>")
        print("finding boundries successfully")
        tasks = [
            (chunk_id,path,start,end)
            for chunk_id,(start,end) in enumerate(zip(boundries[:-1],boundries[1:]))
        ]
        
        with ProcessPoolExecutor(max_workers = max_worker) as executor:
            for local_ct in executor.map(pretokenization,tasks):
                ct.update(local_ct)
        print("a file is scanned")
    print(f"time consumed by pretokenization is {time.perf_counter() - begin}")


    new_ct = Counter()
    for key,value in ct.items():
        new_ct[tuple(bytes([b])  for b in key.encode("utf-8"))] = value

    ct = new_ct

    merges = list()
    start  = time.time()
    epoch = 1
    

    # new version maintaining pair frequency instead of rebuilting repeatedly
    pair_freq = Counter()
    reversed_index = defaultdict(set)
    heap = []
    versions = defaultdict(int)
    for id,(key,value) in enumerate(ct.items()):
        for char1,char2 in zip(key[:-1],key[1:]):
            reversed_index[char1,char2].add(id)
            pair_freq[char1 , char2]+=value
            
    for pair,freq in pair_freq.items():
        heap.append(HeapEntry(freq,pair,0))
    heapq.heapify(heap)

    pretoken_freq_list = list(ct.items())

    total_merges = max_vocabsize - len(vocabulary)

    
    
    while len(vocabulary) < max_vocabsize and pair_freq:
        
        t0 = time.perf_counter()
        chosen = get_best_pair(heap,pair_freq,versions) #shape like (b'a',b'b')
        vocabulary[len(vocabulary)] = chosen[0] + chosen[1]
        merges.append(chosen)
        #then update pair_freq
        t1 = time.perf_counter()
        affected_ids = list(reversed_index[chosen])
        
        #fix bug: b'a',b'b',b'a',b'b' will not count b'ab' b'ab'
        pair_delta = Counter()
        
        for pretoken in affected_ids:
            word,freq = pretoken_freq_list[pretoken]
            old_pairs = Counter(zip(word[:-1],word[1:]))

            #get new word to prepare for the pre-token freq
            new_word = list()
            i = 0
            while i<len(word):
                if i+1 <=len(word)-1 and (word[i],word[i+1]) == chosen:
                    new_word.append(word[i]+word[i+1])
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            new_word = tuple(new_word)
            #compute the new pairs
            new_pairs = Counter(zip(new_word[:-1],new_word[1:]))
            #update pair_freq
            for pair in old_pairs.keys() |new_pairs.keys():
                delta = (new_pairs[pair] - old_pairs[pair])*freq
                
                # if delta == 0:continue
                # pair_freq[pair] += delta
                # if pair_freq[pair] < 0:raise RuntimeError(f"Negative Pair Frequncy: {pair}")
                # if pair_freq[pair] == 0: pair_freq.pop(pair)
                if delta:
                    pair_delta[pair] += delta
                #     update_pair_freq(pair,delta)
            # update the reversed index
            for pair in old_pairs.keys() - new_pairs.keys():
                reversed_index[pair].discard(pretoken)
            
            for pair in new_pairs.keys() - old_pairs.keys():
                reversed_index[pair].add(pretoken)
        
            #save the updated pretoken
            pretoken_freq_list[pretoken] = (new_word, freq)

        for pair,delta in pair_delta.items():
            if delta:
                update_pair_freq(pair_freq,pair,delta)

        print(f"epoch: {epoch} finished")
        epoch +=1

        t2 = time.perf_counter()
        if epoch % 100 == 0:
            print(
                f"epoch={epoch}, "
                f"pairs={len(pair_freq)}, "
                f"affected={len(affected_ids)}, "
                f"max_time={t1-t0:.4f}s, "
                f"update_time={t2-t1:.4f}s"
            )

    print(vocabulary[max(vocabulary,key = lambda x:len(vocabulary[x]))])
    # print(vocabulary)
    end = time.time()
    print(f"time consumed by merge is {end - start}")


    Path("Output/tinystory").mkdir(exist_ok = True,parents=True)
    with open("Output/tinystory/vocab.pkl","wb") as f:
        pickle.dump(vocabulary,f)
    with open("Output/tinystory/merges.pkl","wb") as f:
        pickle.dump(merges,f)


