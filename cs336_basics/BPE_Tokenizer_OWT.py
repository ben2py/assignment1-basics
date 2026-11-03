from .pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter,defaultdict
from typing import BinaryIO
import pickle
from pathlib import Path

import time
from concurrent.futures import ProcessPoolExecutor
import os


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
    root = Path(r"D:\Projects\assignment1-basics\data\openwebtext_txt")

    vocabulary = {
        i:bytes([i]) for i in range(256)
    }
    special_tokens = [b"<|endoftext|>"]
    special_characters = [",",".","!","@","#","$","%","^","&","*","(",")","-","+","=","/",";",":","\'","\"","\n","\t"," ","\r"]
    max_vocabsize = 32000
    num_chunks = 128
    max_worker = 12
    # vocabulary[len(vocabulary)] = b"<|endoftext|>"
    ct = Counter()
    begin= time.perf_counter()
    for file in root.glob("*.txt"):

        path = root / file
        boundries = list()
        with open(path,"rb") as f:
            boundries = find_chunk_boundaries(f,num_chunks,b"\r\n\r\n")
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

    # start1 = time.time()
    # ct = pretokenization("data/data/TinyStoriesV2-GPT4-train.txt",num_chunks,b"<|endoftext|>")

    new_ct = Counter()
    for key,value in ct.items():
        new_ct[tuple(bytes([b])  for b in key.encode("utf-8"))] = value

    ct = new_ct
    # end1 = time.time()
    # print(f"time consumed by pretokenization if {end1 - start1}")
    # print("Succesful Pretokenization")


    merges = list()
    start  = time.time()
    epoch = 1
    # while len(vocabulary) < max_vocabsize:
    #     temp = Counter()
    #     for key,value in ct.items():
    #         for char1,char2 in zip(key[:-1],key[1:]):
    #             temp[(char1,char2)]+=value
    #     if not temp:break
    #     chosen = max(temp,key = lambda pair:(temp[pair],pair))
        
    #     #update ct with the chosen pair
    #     new_ct = Counter()
    #     for key,value in ct.items():
            
    #         cnt = 0
    #         new_key = []
    #         while cnt < len(key):
    #             if (
    #                 cnt + 1 < len(key) and (key[cnt],key[cnt+1])==chosen
    #             ):
    #                 new_key.append(key[cnt]+key[cnt+1])
    #                 cnt +=2
    #             else:
    #                 new_key.append(key[cnt])
    #                 cnt+=1
    #         new_ct[tuple(new_key)] += value
    #     merges.append(chosen)
    #     vocabulary[len(vocabulary)] = chosen[0]+chosen[1]
    #     print(f"epoch {epoch},merge {chosen}, frequency {temp[chosen]}")
    #     ct = new_ct
    #     epoch +=1

    # new version maintaining pair frequency instead of rebuilting repeatedly
    pair_freq = Counter()
    reversed_index = defaultdict(set)
    for id,(key,value) in enumerate(ct.items()):
        for char1,char2 in zip(key[:-1],key[1:]):
            reversed_index[char1,char2].add(id)
            pair_freq[char1 , char2]+=value

    pretoken_freq_list = list(ct.items())

    while len(vocabulary) < max_vocabsize:
        chosen = max(pair_freq,key = lambda x:(pair_freq[x],x)) #shape like (b'a',b'b')
        vocabulary[len(vocabulary)] = chosen[0] + chosen[1]
        merges.append(chosen)
        #then update pair_freq
        
        
        affected_ids = list(reversed_index[chosen])
        # process the pretokens that contain chosen pair
        
        # for pretoken_id in affected_ids:
        #     pretoken_freq_tuple = pretoken_freq_list[pretoken_id] #((b'a',b'c'),10)
        #     pretoken = pretoken_freq_tuple[0]         
            
        #     cnt = 0
        #     new_key = []
        #     while cnt  < len(pretoken):
        #         if cnt+1 < len(pretoken) and (pretoken[cnt],pretoken[cnt+1]) == chosen:
        #             new_key.append(pretoken[cnt] + pretoken[cnt+1])
        #             reversed_index[pretoken[cnt],pretoken[cnt+1]].discard(pretoken_id)
        #             #pair freq 前面已经删掉了
        #             cnt +=2
        #         elif cnt +2 < len(pretoken) and (pretoken[cnt+1],pretoken[cnt+2]) == chosen:
        #             new_key.append(pretoken[cnt])
        #             pair_freq[(pretoken[cnt],pretoken[cnt+1])] -= pretoken_freq_tuple[1]
        #             pair_freq[(pretoken[cnt],pretoken[cnt+1]+pretoken[cnt+2])] += pretoken_freq_tuple[1]
        #             reversed_index[pretoken[cnt],pretoken[cnt+1]+pretoken[cnt+2]].add(pretoken_id)
        #             reversed_index[pretoken[cnt],pretoken[cnt+1]].discard(pretoken_id)
        #             cnt+=1
                    
        #         elif cnt>1 and (pretoken[cnt-2],pretoken[cnt-1]) == chosen:
        #             new_key.append(pretoken[cnt])
        #             pair_freq[(pretoken[cnt-1],pretoken[cnt])] -= pretoken_freq_tuple[1]
        #             pair_freq[(pretoken[cnt-2]+pretoken[cnt-1],pretoken[cnt])] += pretoken_freq_tuple[1]
        #             reversed_index[pretoken[cnt-2]+pretoken[cnt-1],pretoken[cnt]].add(pretoken_id)
        #             reversed_index[pretoken[cnt-1],pretoken[cnt]].discard(pretoken_id)
        #             cnt+=1
                    
        #         else:
        #             new_key.append(pretoken[cnt])
        #             cnt+=1
        #     pretoken_freq_list[pretoken_id] = (tuple(new_key),pretoken_freq_tuple[1])

        #fix bug: b'a',b'b',b'a',b'b' will not count b'ab' b'ab'
        for pretoken in affected_ids:
            word,freq = pretoken_freq_list[pretoken]
            old_pairs = Counter(zip(word[:-1],word[1:]))
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
            #
            for pair in old_pairs.keys() |new_pairs.keys():
                delta = (new_pairs[pair] - old_pairs[pair])*freq
                if delta == 0:continue
                pair_freq[pair] += delta
                if pair_freq[pair] < 0:raise RuntimeError(f"Negative Pair Frequncy: {pair}")
                if pair_freq[pair] == 0: pair_freq.pop(pair)
            # update the reversed index
            for pair in old_pairs.keys() - new_pairs.keys():
                reversed_index[pair].discard(pretoken)
            
            for pair in new_pairs.keys() - old_pairs.keys():
                reversed_index[pair].add(pretoken)

            #save the updated pretoken
            pretoken_freq_list[pretoken] = (new_word, freq)

        print(f"epoch: {epoch} finished")
        epoch +=1

    print(vocabulary[max(vocabulary,key = lambda x:len(vocabulary[x]))])
    # print(vocabulary)
    end = time.time()
    print(f"time consumed by merge is {end - start}")


    # Path("Output").mkdir(exist_ok = True)
    # with open("Output/vocab.pkl","wb") as f:
    #     pickle.dump(vocabulary,f)
    # with open("Output/merges.pkl","wb") as f:
    #     pickle.dump(merges,f)

# from collections import Counter   
# ct1 = Counter([("a","b"),("a","c")])
# ct2 = Counter([("a","c"),("a","b"),("a","c")])

# print(ct1.most_common(1))
# print(ct2.most_common(1))

# temp = sorted(ct2,key = lambda pair:(ct2[pair],pair),reverse = True)
# print(temp)
