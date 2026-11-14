import pickle

if __name__ == "__main__":

    # with open ("Output/merges.pkl","rb") as f:
    #     merges = pickle.load(f)

    # print(merges)

    with open("Output/openwebtext/vocab.pkl","rb") as f:
        vocabulary = pickle.load(f)

    token_id = max(vocabulary,key = lambda x:len(vocabulary[x]))
    print(token_id)
    print(vocabulary[token_id])

    print(f"vocab length: {len(vocabulary)}")
    for i in range(20):

        print(vocabulary[i+256])
    