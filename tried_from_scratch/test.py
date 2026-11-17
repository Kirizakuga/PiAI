import pickle
path = r"C:\Users\karus\Documents\project\label\nanoGPT\data\karus\meta.pkl"
with open(path, "wb") as f:
    pickle.dump({"vocab_size": 4000}, f)
print("saved", path)