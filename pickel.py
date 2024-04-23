import pickle

with open('training_pickle/training_x.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data[10:])
with open('training_pickle/training_y.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data[10:])