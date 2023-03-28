import numpy as np

def make_folds(X, T, k, shuffle=False):
    row_indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(row_indices)
    folds = {}
    classes = np.unique(T)
    for c in classes:
        class_indices = row_indices[np.where(T[row_indices, :] == c)[0]]
        n_in_class = len(class_indices)
        n_each = int(n_in_class / k)
        starts = np.arange(0, n_each * k, n_each)
        stops = starts + n_each
        stops[-1] = n_in_class
        folds[c] = [class_indices, starts, stops]
    return folds

def rows_in_fold(folds, fold_i):
    all_rows = []
    for c, rows in folds.items():
        class_rows, starts, stops = rows
        all_rows += class_rows[starts[fold_i]:stops[fold_i]].tolist()
    return all_rows

def rows_in_folds(folds, fold_i_s):
    all_rows = []
    for fold_i in fold_i_s:
        all_rows += rows_in_fold(folds, fold_i)
    return all_rows

def stratified_k_fold_partitions(X, T, k, shuffle=True):
    '''
    Example:

    X = np.array([f'cat{i+1}' for i in range(10)] +
                 [f'dog{i+1}' for i in range(5)]).reshape(-1, 1)
    T = np.array([0] * 10 + [1] * 5).reshape(-1, 1)

    print(f'T {T.shape} cats/dogs {np.sum(T==0)/np.sum(T==1):.2f}')

    reps = 0
    for Xtrain, Ttrain, Xtest, Ttest in stratified_k_fold_partitions(X, T, k=4, shuffle=True):
        reps += 1
        print()
        print(''='*10, 'Repetition', reps)
        print('Xtrain', Xtrain.shape)
        print(Xtrain)
        print(f'Ttrain {Ttrain.shape} cats/dogs {np.sum(Ttrain==0)/np.sum(Ttrain==1):.2f}')
        print(Ttrain)
        print('Xtest', Xtest.shape)
        print(Xtest)
        print(f'T {Ttest.shape} cats/dogs {np.sum(Ttest==0)/np.sum(Ttest==1):.2f}')
        print(Ttest)
    '''

    folds = make_folds(X, T, k, shuffle)

    for test_fold in range(k):
        train_folds = np.setdiff1d(range(k), [test_fold])
        rows = rows_in_fold(folds, test_fold)
        Xtest = X[rows, :]
        Ttest = T[rows, :]
        rows = rows_in_folds(folds, train_folds)
        Xtrain = X[rows, :]
        Ttrain = T[rows, :]
        yield Xtrain, Ttrain, Xtest, Ttest  # HERE IS THE GENERATOR STATEMENT

if __name__ == '__main__':

    X = np.array([f'cat{i+1}' for i in range(10)] +
                 [f'dog{i+1}' for i in range(5)]).reshape(-1, 1)
    T = np.array([0] * 10 + [1] * 5).reshape(-1, 1)

    reps = 0
    for Xtrain, Ttrain, Xtest, Ttest in stratified_k_fold_partitions(X, T, k=4, shuffle=True):
        reps += 1
        print()
        print('='*10, 'Repetition', reps)
        print('Xtrain', Xtrain.shape)
        print(Xtrain)
        print(f'Ttrain {Ttrain.shape} cats/dogs {np.sum(Ttrain==0)/np.sum(Ttrain==1):.2f}')
        print(Ttrain)
        print('Xtest', Xtest.shape)
        print(Xtest)
        print(f'Ttest {Ttest.shape} cats/dogs {np.sum(Ttest==0)/np.sum(Ttest==1):.2f}')
        print(Ttest)
