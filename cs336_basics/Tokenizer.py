import pickle
from .pretokenization_example import find_chunk_boundaries
import regex as re
from pathlib import Path
from typing import Iterable,Iterator
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
class Tokenizer:
    def __init__(self,vocab,merges,special_tokens = None):
        self.vocab = vocab
        self.special_tokens = special_tokens or []
        if special_tokens:
            for special_token in self.special_tokens:
                self.vocab[len(self.vocab)] = special_token.encode("utf-8")
        self.merges =merges
        self.reversed_dict = {value:key for key,value in self.vocab.items()}
        self.merge_rank = {
                pair:rank 
                for rank,pair in enumerate(self.merges)
            }
        self.max_vocabsize = len(self.vocab)
        self.PAT = re.compile(re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""))
        if self.special_tokens:
            special_pattern = "|".join(
                re.escape(special_token) for special_token in sorted(self.special_tokens,key=len,reverse=True)
            )
            self.special_pattern = re.compile(special_pattern)
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)

        return cls(vocab,merges,special_tokens)
    @lru_cache(maxsize=100_000)
    def _encode_pretoken(self,word):
        elements = [bytes([b]) for b in word.encode("utf-8")]
                    
        if len(elements) == 1:
            return elements
        
        # while True:
        #     if len(elements) == 1:
        #         return elements
        #     # new_word = list()
            
        #     pair_list = [(char1,char2) for char1,char2 in zip(elements[:-1],elements[1:])]
            
        #     prio_list = [self.merge_rank.get(pair,self.max_vocabsize+1) for pair in pair_list]
            
        #     idx = min(range(len(prio_list)),key=lambda i:prio_list[i])
        #     if all([i==self.max_vocabsize+1 for i in prio_list ]): break
        while len(elements) >1:
            best_rank = float("inf")
            best_idx = -1

            for i in range(len(elements)-1):
                pair = (elements[i],elements[i+1])
                
                if self.merge_rank.get(pair,float("inf")) < best_rank:
                    best_rank = best_rank
                    best_idx = i
            # for i in range(len(elements)):
            #     if i == idx:new_word.append(elements[i]+elements[i+1])
            #     elif i!= idx +1:
            #         new_word.append(elements[i])
            # elements = new_word
            if best_idx == -1: break
            idx = best_idx
            elements[idx:idx+2] = [elements[idx]+elements[idx+1]]
            # print(elements)
        return elements


    def encode(self, text: str) -> list[int]:
        
        encoded = []
        # for special_token in self.special_tokens:
        
        if self.special_pattern:

            parts = re.split(f"({self.special_pattern})",text)
            # re.split(special_pattern),会去除掉 special_pattern
            # re.split(f"({special_pattern})") 会保留 special_pattern

        else:
            parts = [text]
        
        # print(parts)
        for part in parts:
            
            if part in self.special_tokens:
                encoded.append(self.reversed_dict[part.encode("utf-8")])
            else:
                for word in self.PAT.findall(part):
                    # print(PAT.findall(text))
                    elements = self._encode_pretoken(word)
                   
                    encoded.extend([self.reversed_dict[value] for value in elements])

        return encoded

           
                            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # iterable is a file handler
        for text in iterable:
            yield from self.encode(text) # 委托迭代


    def decode(self, ids: list[int]) -> str:
        res = b""
        for id in ids:
            res += self.vocab[id]

        return res.decode("utf-8",errors = "replace")

if __name__ == "__main__":
    root = Path("Output/openwebtext")
    t = Tokenizer.from_files(root/"vocab.pkl",root/"merges.pkl",special_tokens=["<|endoftext|>"])

    print(t.decode(t.encode("<|endoftext|>")))
    # print(t.decode(t.encode("你好")))
    # data_root = Path("data/openwebtext_txt")
    # for i in t.encode_iterable(data_root/"train-00000-of-00080.txt"):
    #     print(i)
    #     break

    

