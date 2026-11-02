import pickle

if __name__ == "__main__":

    # with open ("Output/merges.pkl","rb") as f:
    #     merges = pickle.load(f)

    # print(merges)

    with open("Output/vocab.pkl","rb") as f:
        vocabulary = pickle.load(f)

    token_id = max(vocabulary,key = lambda x:len(vocabulary[x]))
    print(vocabulary[token_id])
    