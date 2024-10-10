from model import*
from measrmnt_indices import*

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from skrebate import ReliefF
import matplotlib.pyplot as plt

def RFE_Selection( X, y, model):
    rfecv = RFECV(estimator=model, step=1, cv=10)
    rfecv.fit(X, y)
    return rfecv

def ReliefF_Selection( X, y):
    fs = ReliefF()
    fs.fit(X, y)
    return fs

def Corr(dataFr):
    cor = dataFr.corr()
    cor_target = abs(cor["Hb (g/dL)"])
    return cor_target

def selected_index_RFE(data, feature_value, criteria, f_length):
    columns_names = data.columns.values
    selected_columns_names = []
    get_best_ind = []
    for i in range(f_length):
        if feature_value[i] == criteria:
            get_best_ind.append(i)
    print(len(get_best_ind))      
    selected_columns_names =columns_names[get_best_ind]
    print(selected_columns_names)
    return selected_columns_names, get_best_ind

def selected_feature_and_index(data, feature_value, criteria, f_length):
    columns_names = data.columns.values
    selected_columns_names = []
    get_best_ind = []
    for i in range(f_length):
        if feature_value[i] >= criteria:
            get_best_ind.append(i)
    print(len(get_best_ind))      
    selected_columns_names =columns_names[get_best_ind]
    print(selected_columns_names)
    return selected_columns_names, get_best_ind
   
def select_three_or_four_FSM(data, f_length, fsm_1, fsm_2, fsm_3, fsm_4):
    columns_names = data.columns.values
    selected_columns_names = []
    get_best_ind = []
    final_list = fsm_1 + fsm_2 + fsm_3 + fsm_4   
    # occurrence = {item: final_list.count(item) for item in final_list}
    for i in set(final_list):
        if final_list.count(i) >= 4:
            get_best_ind.append(i)
    print(len(get_best_ind))      
    selected_columns_names =columns_names[get_best_ind]
    print(selected_columns_names)
    return selected_columns_names, get_best_ind  
