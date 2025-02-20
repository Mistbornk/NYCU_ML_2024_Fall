import numpy as np
from scipy.optimize import linear_sum_assignment

def init_lambda():
    '''
    lambda[k]= prior of class k
    sum(lambda)=1
    :return: (10,1) matrix
    '''
    re = np.random.rand(10)
    re = re / np.sum(re)
    return re

def init_PixVal():
    '''
    P[i,j]: pixel value==1 prob in class i's jth feature distribution
    :return: (10,784) matrix
    '''
    re = np.random.uniform(low=0.25, high=0.9, size=(10, 784))
    return re

def Update_Posterior(train_x, Lambda, Distribution):
    '''
    update posterior using log likelihood
    :param X_train: (60000,784) 0-1 uint8 matrix
    :param Lambda: (10,1)
    :param Distribution: (10,784)
    :return: (60000,10)
    '''
    Distribution_complement = 1 - Distribution
    W=np.zeros((60000,10))

    for i in range(60000):
        for j in range(10):
            W[i,j]=np.prod(train_x[i]*Distribution[j]+(1-train_x[i])*Distribution_complement[j])

    #add prior
    W = W*Lambda.reshape(1,-1)
    sums = np.sum(W, axis=1, keepdims=True)
    sums[sums == 0] = 1  # 避免除以 0
    W /= sums
    
    return W

def Update_Lambda(W):
    '''
    :param W: (60000,10)
    :return: (10,1)
    '''
    Lambda = np.sum(W, axis=0)
    Lambda = Lambda/60000
    return Lambda

def Update_Distribution(train_x, W):
    '''
    A.T@W -> normalized,transpose -> concate with 1-complement
    :param train_x: (60000,784)
    :param W: (60000,10)
    :return: (10,784)
    ''' 
    sums = np.sum(W, axis=0)
    sums[sums==0] = 1
    W_normalized = W/sums
    PixVal = train_x.T @ W_normalized
    return PixVal.T

def get_pixvalueProb_discrete(train_x, train_y):
    '''
    get pixvalue_prob conditional on class & dim
    :param train_x: (60000,784) 0-1 matrix
    :param train_y: (60000,)
    :return: (10,784) probability matrix of pixelValue==1
    '''
    labels = np.zeros(10)
    for label in train_y:
        labels[label] += 1

    distribution=np.zeros((10,784))
    for i in range(60000):
        clss = train_y[i]
        for j in range(784):
            if train_x[i,j]==1:
                distribution[clss,j]+=1

    #normalized
    distribution = distribution / labels.reshape(-1,1)

    return distribution 

def plot_discrete(Distribution):
    '''
    :param Distribution: (10,784)
    :return:
    '''
    for c in range(10):
        print(f'labeled class {c}:')
        for i in range(28):
            for j in range(28):
                print('+' if Distribution[c,i*28+j]>0.35 else 0,end=' ')
            print()
        print()
        print()    

def distance(a, b):
    '''
    :param a: (784)
    :param b: (784)
    :return: euclidean distance between a and b
    '''
    return np.linalg.norm(a-b)

def hungarian_algo(Cost):
    '''
    match GT to our estimate
    :param Cost: (10,10)
    :return: (10) column index
    '''
    row_idx, col_idx=linear_sum_assignment(Cost)
    return col_idx

def perfect_matching(ground_truth, estimate):
    '''
    matching GT_distribution to estimate_distribution by minimizing the sum of distance
    :param ground_truth: (10,784)
    :param estimate: (10,784)
    :return: (10)
    '''
    Cost=np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            Cost[i,j]=distance(ground_truth[i],estimate[j])

    classes_order = hungarian_algo(Cost)

    return classes_order

def plot(Distribution, classes_order, threshold):
    '''
    plot each classes expected pattern
    :param Distribution: (10,784)
    :param classes_order: (10)
    :param threshold: value between 0.0~1.0
    :return:
    '''
    Pattern = np.asarray(Distribution>threshold,dtype='uint8')
    for i in range(10):
        print('labeled class {}:'.format(i))
        plot_pattern(Pattern[classes_order[i]])
    return

def confusion_matrix(real, predict, classes_order):
    '''
    :param real: (60000)
    :param predict: (60000)
    :param classes_order: (10)
    :return:
    '''
    #sorted(classes_order)
    #for c in classes_order:
    #    # 將 real 和 predict 中對應於類別 c 的情況轉換成布爾向量
    #    real_is_c = (real == c)
    #    predict_is_c = (predict == c)

    #    # 使用向量化方式計算 TP, FN, FP, TN
    #    TP = np.sum(real_is_c & predict_is_c)
    #    TN = np.sum(~real_is_c & ~predict_is_c)
    #    FP = np.sum(~real_is_c & predict_is_c)
    #    FN = np.sum(real_is_c & ~predict_is_c)

    #    # 繪製混淆矩陣
    #    plot_confusion_matrix(c, TP, FN, FP, TN)
    classes_order = classes_order[np.argsort(classes_order)]
    for i in range(10):
        c=classes_order[i]
        TP,FN,FP,TN=0,0,0,0
        for i in range(60000):
            if real[i]!=c and predict[i]!=c:
                TN+=1
            elif real[i]==c and predict[i]==c:
                TP+=1
            elif real[i]!=c and predict[i]==c:
                FP+=1
            else:
                FN+=1
        plot_confusion_matrix(c,TP,FN,FP,TN)

def print_error_rate(count, real, predict, classes_order):
    '''
    :param count: int
    :param real: (60000)
    :param predict: (60000)
    :param classes_order: (10)
    :return:
    '''
    print('Total iteration to converge: {}'.format(count))
    real_transform=np.zeros(60000)
    for i in range(60000):
        real_transform[i]=classes_order[real[i]]
    error=np.count_nonzero(real_transform-predict)
    print('Total error rate: {}'.format(error/60000))



def plot_confusion_matrix(c, TP, FN, FP, TN):
    print('------------------------------------------------------------')
    print()
    print('Confusion Matrix {}:'.format(c))
    print('\t\t\t  Predict number {} Predict not number {}'.format(c, c))
    print('Is number {}\t\t{}\t\t\t\t{}'.format(c,TP,FN))
    print('Isn\'t number {}\t\t{}\t\t\t\t{}'.format(c,FP,TN))
    print()
    print('Sensitivity (Successfully predict number {}): {:.5f}'.format(c,TP/(TP+FN)))
    print('Specificity (Successfully predict not number {}): {:.5f}'.format(c,TN/(TN+FP)))
    print()

def plot_pattern(pattern):
    '''
    :param pattern: (784)
    :return:
    '''
    for i in range(28):
        for j in range(28):
            print("+" if pattern[i*28+j] == 1 else 0,end=' ')
        print()
    print()
    print()
    return