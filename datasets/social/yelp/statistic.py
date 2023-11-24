import pickle

def load_pkl(file):
    with open(file, 'rb') as fs:
        data = pickle.load(fs)
    return data

if __name__=="__main__":
    category = load_pkl('category.pkl')
    print(category)

    trn_mat = load_pkl('trn_mat.pkl')
    print(len(trn_mat.data))