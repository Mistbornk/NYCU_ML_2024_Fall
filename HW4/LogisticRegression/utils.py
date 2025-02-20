import numpy as np
import matplotlib.pyplot as plt

def univariate_gaussian_data_generator(mean, variance):
    return (np.sum(np.random.uniform(0, 1, 12)) - 6) * np.sqrt(variance) + mean

def sampling(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2):
    '''
    :param N: sampling N data points
    :param mx: x mean
    :param my: y mean
    :param vx: x variance
    :param vy: y variance
    :return: (N, 2) shape matrix
    '''
    D1 = np.zeros((N, 2))
    D2 = np.zeros((N, 2))
    for i in range(N):
        D1[i, 0] = univariate_gaussian_data_generator(mx1, vx1)
        D1[i, 1] = univariate_gaussian_data_generator(my1, vy1)
        D2[i, 0] = univariate_gaussian_data_generator(mx2, vx2)
        D2[i, 1] = univariate_gaussian_data_generator(my2, vy2)

    return D1, D2

def init_phi(D1, D2):
    phi = np.ones((2 * len(D1), 3))
    phi[:, 1:] = np.vstack((D1, D2))
    return phi

def init_group(N):
    group = np.zeros((2*N, 1))
    group[N:] = np.ones((N, 1))
    return group

def predict(phi, omega):
    '''
    predict whether is class0 or class1
    :param A: (2N,3) shape matrix
    :param w: (3,1) shape matrix
    :return: (2N,1) shape matrix
    '''
    N=len(phi)
    group_predict = np.empty((N, 1))
    for i in range(N):
        group_predict[i]=0 if phi[i]@omega<0 else 1

    return group_predict

def confusion_matrix(phi, group, group_predict):
    '''
    let class0 be positive, class1 be negative
    ----------
    | TP  FN |  <= confusion matrix by HW
    | FP  TN |
    ----------
    :param A: (2N,3) shape matrix
    :param b: (2N,1) shape matrix
    :param b_predict: (2N,1) shape matrix
    :return: (confusion_matix, points to be class0, points to be class1)
    '''
    doubleN = len(phi)
    group_concate = np.hstack((group, group_predict))
    TP = FP = FN = TN = 0
    for pair in group_concate:
        if pair[0]==pair[1]==1:
            TP+=1
        elif pair[0]==pair[1]==0:
            TN+=1
        elif pair[0]==1 and pair[1]==0:
            FP+=1
        else:
            FN+=1
    matrix=np.empty((2, 2))
    matrix[0,0],matrix[0,1],matrix[1,0],matrix[1,1] = TP, FN, FP, TN

    D1_predict=[]
    D2_predict=[]
    for i in range(doubleN):
        if group_predict[i]==0:
            D1_predict.append(phi[i,1:])
        else:
            D2_predict.append(phi[i,1:])

    return (matrix,np.array(D1_predict), np.array(D2_predict))

def print_omega(omega):
    print('w:')
    print(omega[0])
    print(omega[1])
    print(omega[2])

def print_confusion_matrix(matrix):
    print('Confusion Matrix:')
    print('               Predict cluster 1  Predict cluster 2')
    print('Is cluster 1        {:.0f}               {:.0f}       '.format(matrix[0,0],matrix[0,1]))
    print('Is cluster 2        {:.0f}               {:.0f}       '.format(matrix[1,0],matrix[1,1]))
    print()
    print('Sensitivity (Successfully predict cluster 1): {}'.format(matrix[0,0]/(matrix[0,0]+matrix[1,0])))
    print('Specificity (Successfully predict cluster 2): {}'.format(matrix[0,0]/(matrix[0,0]+matrix[0,1])))

def plot(D1, D2, axs, title):
    axs.plot(D1[:,0], D1[:,1],'ro')
    axs.plot(D2[:,0], D2[:,1],'bo')
    axs.set_title(title)

def print_results(N, phi, group, gd_omega, nm_omega, D1, D2):
    """
    Print the results and draw the graph
    :param num_of_points: number of data points
    :param phi: Φ matrix
    :param group: group of each data point
    :param gd_weight: weight vector omega from gradient descent
    :param nm_weight: weight vector omega from Newton's method
    :return: None
    """
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))

    plot(D1, D2, axs[0],'Ground truth')

    print('Gradient descent:\n')
    group_predict = predict(phi, gd_omega)
    matrix, D1_predict, D2_predict = confusion_matrix(phi, group, group_predict)
    print_omega(gd_omega)
    print_confusion_matrix(matrix)
    plot(D1_predict, D2_predict, axs[1], 'Gradient descent')
    
    print('\n----------------------------------------')
    print("Newton's method:")
    group_predict = predict(phi, nm_omega)
    matrix, D1_predict, D2_predict = confusion_matrix(phi, group, group_predict)
    print_omega(nm_omega)
    print_confusion_matrix(matrix)
    plot(D1_predict, D2_predict, axs[2],'Newton\'s method')

    plt.tight_layout()
    plt.show()
