from sklearn import neighbors
from data_util import DataUtils
import datetime
import numpy as np

def onehot(data):
    onehot_encoded = []
    for value in data:
        num = [0 for _ in range(10)]
        num[value] = 1
        onehot_encoded.append(num)
    return onehot_encoded

def main():
    trainfile_X = "train-images-idx3-ubyte"
    trainfile_y = "train-labels-idx1-ubyte"
    testfile_X = "t10k-images-idx3-ubyte"
    testfile_y = "t10k-labels-idx1-ubyte"
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y_original = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y_original = DataUtils(testfile_y).getLabel()
    train_y_onehot = onehot(train_y_original)
    test_y_onehot = onehot(test_y_original)
    np.savetxt("train_X.txt", train_X)
    np.savetxt("train_y_original.txt", train_y_original)
    np.savetxt("test_X.txt", test_X)
    np.savetxt("test_y_original.txt", test_y_original)
    np.savetxt("train_y_onehot.txt", train_y_onehot)
    np.savetxt("test_y_onehot.txt", test_y_onehot)

    return

if __name__ == "__main__":
    main()
