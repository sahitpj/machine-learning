'''
Assuming all the data is in the form of a DataFrame,

However input to trees are in the form of row indecies. 
'''
import math, random

def K_fold_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def normal_split(dataset, param):
    dataset_split = list()
    dataset_copy = list(dataset)
    train_size = int(math.floor(len(dataset)*param))
    test_size = len(dataset)-train_size
    validate_size = int(math.floor(train_size*0.3))
    train_size = train_size-validate_size
    assert(train_size+validate_size+test_size <= len(dataset))
    fold = list()
    while len(fold) < train_size:
        index = random.randrange(len(dataset_copy))
        k = dataset_copy.pop(index)
        fold.append(index)
    dataset_split.append(fold)
    fold = list()
    while len(fold) < validate_size:
        index = random.randrange(len(dataset_copy))
        k = dataset_copy.pop(index)
        fold.append(index)
    dataset_split.append(fold)
    fold = list()
    while len(fold) < test_size:
        index = random.randrange(len(dataset_copy))
        k = dataset_copy.pop(index)
        fold.append(index)
    dataset_split.append(fold)
    return dataset_split  #in the order of train, validate, test
