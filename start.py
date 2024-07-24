import os
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cvxopt

from importlib import reload
import methods
reload(methods)
from methods import *

seed()


def main():
    
    # Load data and preprocess
    print('\nData Loading and Preprocessing...')
    t0 = time.time()
    Xtr, Xte, Ytr = load_data()
    Xtr_features_3D, Xte_features_3D = data_features_3D(Xtr, Xte)
    X_train_features_3D, X_val_features_3D, Y_train_features_3D, Y_val_features_3D = stratified_train_test_split(Xtr_features_3D, Ytr)
    t1 = time.time() - t0
    print(f'---> Done in {t1:.2f} s\n')
    
    # KRR 3D
    print('KRR Training and Prediction...')
    t0 = time.time()
    kernels_krr_3D, best_lambdas_3D = parameters_krr(X_train_features_3D, X_val_features_3D, Y_train_features_3D, Y_val_features_3D)
    lam_3D, kernel_3D = best_lambdas_3D[2], kernels_krr_3D[2]
    application_krr(Xtr_features_3D, Xte_features_3D, Ytr, lam_3D, kernel_3D)
    t1 = time.time() - t0
    print(f'---> Done in {t1:.2f} s\n')
    
    # SVM 3D
    print('SVM Training and Prediction... (a bit long)')
    t0 = time.time()
    kernel_SVM = GaussianKernel(sigma=0.5995)
    C_SVM = parameters_svm(X_train_features_3D, X_val_features_3D, Y_train_features_3D, Y_val_features_3D, kernel=kernel_SVM)
    application_svm(Xtr_features_3D, Xte_features_3D, Ytr, C_SVM, kernel_SVM)
    t1 = time.time() - t0
    t1_min, t1_sec = divmod(t1, 60)
    print(f'---> Done in {t1_min:.0f} min {t1_sec:.0f} s\n')
    
    
if __name__ == "__main__":
    main()