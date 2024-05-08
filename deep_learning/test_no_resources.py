import no_resources
import numpy as np
import unittest

def test_chunk():
    data_path = "/Users/Tighe_Clough/Desktop/Programming/Projects/i-spy-tickers/data/examples_test.csv"
    chunky = no_resources.Chunk(data_path, chunk_size=10)
    chunky_iter1 = chunky.generate(tdt_sizes=(.6,.2,.2), make_sparse=True, sparse_dim=1000)
    chunky_iter2 = chunky.generate(tdt_sizes=(.6,.2,.2))

    for data in chunky_iter1:
        X_tr, y_tr, X_dev, y_dev, X_test, y_test = data

        print(X_tr)
        # print(y_tr)
        # print()
        # print(X_dev)
        # print(y_dev)
        # print()
        # print(X_test)
        # print(y_test)

        print("next chunk")

    for data in chunky_iter2:
        X_tr, y_tr, X_dev, y_dev, X_test, y_test = data

        print(X_tr)
        # print(y_tr)
        # print()
        # print(X_dev)
        # print(y_dev)
        # print()
        # print(X_test)
        # print(y_test)

        print("next chunk")

def test_oha():
    idx_array  = [[1],[0,2],[2]]
    oha_dict = {x:y for x,y in enumerate(idx_array)}
    
    test_oha_validation(oha_dict)
    test_mat_mul(oha_dict, idx_array)
    test_get_item(oha_dict)

def test_oha_validation(oha_dict):
    no_resources.OneHotArray(shape=(3,1), oha_dict=oha_dict)

def test_mat_mul(oha_dict, idx_array):
    sparse_matrix = np.array([[0,1,0],[1,0,1],[0,0,1]])

    test = np.arange(9).reshape(3,3)

    oha_idx = no_resources.OneHotArray(shape=(3,3), idx_array=idx_array)
    oha = no_resources.OneHotArray(shape=(3,3), oha_dict=oha_dict)

    assert oha_idx.idx_rel == oha.idx_rel

    ground1 = sparse_matrix @ test
    produced1 = oha @ test

    assert(np.allclose(ground1,produced1))

    # transpose and matmul
    ground2 = sparse_matrix.T @ test
    produced2 = oha.T @ test

    assert(np.allclose(ground2,produced2))

    # need to add to numpy matmul in herit super().__matmul()__ after

    # ground3 = test @ sparse_matrix

    # produced3 = test @ oha

    # print(ground3)
    # print(produced3)
    

def test_get_item(oha_dict):
    oha2 = no_resources.OneHotArray(shape=(4,4), oha_dict=oha_dict)

    sliced = oha2[[0,2,3]]
    assert sliced.idx_rel == {0:[1], 1:[2]}
    assert sliced.shape == (3,4)
    
def main():
    #test_chunk()
    test_oha()

if __name__ == "__main__":
    main()