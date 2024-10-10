import math
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from keras.models import load_model 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import pairwise_distances
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, AdaBoostRegressor
def RMSE(targets,predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import seaborn as sns;
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import seaborn as sns;
from sklearn import metrics
import math
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from keras.models import load_model 
from sklearn import ensemble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns;
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import seaborn as sns;
#from utils import *
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from scipy import stats
from scipy.special import boxcox, inv_boxcox
from sklearn import linear_model
import xgboost 

# Import other files

from graph import*
from model import*
from measrmnt_indices import* 
from Feature_selection import*

dataFr = pd.read_csv("H:/Glucose/Preprocessed File/preprocessed_PPG-34.csv")
dataFr.drop(dataFr.columns[[0, 1, 40]], axis=1, inplace=True) 
# dataFr.drop(dataFr.columns[[0, 1, 38, 40]], axis=1, inplace=True) #GL
dataFr.head()
dataFr.shape
df = dataFr


dataFr = dataFr.drop(labels=[26, 67], axis=0)


# print(dataFr[dataFr['Sex(M/F)']==1]['Sex(M/F)'].count())

Xorg = dataFr.to_numpy()
scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorg)
Xmeans = scaler.mean_
Xstds = scaler.scale_
y = Xscaled[:, 36]
X = Xscaled[:, 0:36]



#RFE selection
columns_names = dataFr.columns.values
rfe = RFE_Selection(X, y, RFR()) 
print("Optimal number of features : %d" % rfe.n_features_)
selected_columns_names_rfe, get_best_ind_rfe = selected_index_RFE(dataFr, rfe.ranking_, 1, 36)
X_rfe= X[:, get_best_ind_rfe]
print(X_rfe.shape)
# df.drop(dataFr.columns[36], axis=1, inplace=True)
# rfe.support_rfecv_df = pd.DataFrame(rfe.ranking_,index=df.columns,columns=['Rank']).sort_values(by='Rank',ascending=True)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(6,3))
# fig.set_facecolor("white")
# ax.set_xlabel("Number of features selected")
# ax.set_ylabel("Cross validation score($R^2$)")
# plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
# plt.show()


#RReliefF Feature selection
relief = ReliefF_Selection(X, y) 
print(relief.feature_importances_)
# print(relief.top_features_) #index of the features sorted according to feature importance
# for feature_name, feature_score in zip(df.columns, relief.feature_importances_):
#     print(feature_name, '\t', feature_score)

selected_columns_names_relief, get_best_ind_relief = selected_feature_and_index(dataFr, relief.feature_importances_, 0.02, 36)
X_relief = X[:, get_best_ind_relief]
print(X_relief.shape)


#Random Forest Feature selection
rfr = RFR()
rfr.fit(X, y)
print(rfr.feature_importances_)

selected_columns_names_rfr, get_best_ind_rfr = selected_feature_and_index(dataFr, rfr.feature_importances_, 0.01, 36)
X_rfr = X[:, get_best_ind_rfr]
print(X_rfr.shape)


# #Forward Base Feature Selection 

# from sklearn.feature_selection import SequentialFeatureSelector
# sfs = SequentialFeatureSelector( RFR(), cv=10)
# sfs.fit(X, y)
# print(sfs.get_support())
# print(sfs.ranking_)



#Pearson Correlation Base
# Correlation with output variable
cor_target =Corr(dataFr)
selected_columns_names_cor, get_best_ind_cor = selected_feature_and_index(dataFr, cor_target, 0.1, 36)
X_cor = X[:, get_best_ind_rfr]
print(X_cor.shape)


#Select the features that selected by four FSM
selected_columns_names_final, get_best_ind_final = select_three_or_four_FSM(dataFr, 36, get_best_ind_rfr,\
                                        get_best_ind_relief, get_best_ind_rfr, get_best_ind_cor)

X_final = X[:, get_best_ind_final]


# For MIC Feature Selection 
get_best_ind_final_Gl = [2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 23, 24, 25, 26, 29, 31, 33]
X_final_Gl = X[:, get_best_ind_final_Gl]


get_best_ind_final_Hb = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 17, 18, 20, 22, 23, 24, 25, 26, 29, 31, 34, 35]
X_final_Hb = X[:, get_best_ind_final_Hb]



import os
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers 
import random
import pandas as pd
import numpy as np



X = X[:, get_best_ind_final_Hb]

model = DNN(X)
model.summary()

# keras_model_path = 'H:/Glucose/Preprocessed File/GL Model'

n_splits = 10
kf = 0
score = []
std_r2 = []
std_mae = []
cv_set = np.repeat(-1.,X.shape[0])
skf = KFold(n_splits = n_splits ,shuffle=True, random_state=42)
for train_index,test_index in skf.split(X, y):
    x_train,x_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    if x_train.shape[0] != y_train.shape[0]:
        raise Exception()
    # model.fit(x_train,y_train)

    # --------------------------------------
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5) 

    model.fit(x_train,y_train, epochs=100, batch_size=32,
                        shuffle=True,  callbacks=[callback],  validation_split=0, verbose=1)
    # model.save("H:/Glucose/Preprocessed File/GL Model", 'DNN_model' + str(kf) + '.h5')  
    # model.save(keras_model_path) #To Save in Saved_model format
    # model.save('DNN_model_Gl' + str(kf) + '.h5') #To save model in H5 or HDF5 format   
    predicted_y =  model.predict(x_test)
    LOG_INFO(f"Individual R = {pearsonr(y_test, predicted_y)}", mcolor="green")
    print("R: " + str(pearsonr(y_test, predicted_y)))
    cv_set[test_index] = predicted_y[:, 0]
    kf+=1


# import tensorflow as tf
# h5_model = tf.keras.models.load_model("DNN_model_Gl0.h5") # loading model in h5 format
# h5_model.summary()
# saved_m = tf.keras.models.load_model("saved_model/my_model") #loading model in saved_model format
# saved_m.summary()


# print("R^2 (Avg. +/- Std.) is  %0.3f +/- %0.3f" %(np.mean(R_2),np.std(R_2)))    
# print("MAE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mae),np.std(mae)))    
# print("MSE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mse),np.std(mse)))   
# print("RMSE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.sqrt(np.mean(mse)),np.sqrt(np.std(mse))))
# print("MAPE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mape),np.std(mape)))

## For Get real values of target label
yy = (y * Xstds[36]) + Xmeans[36]
cv_sety = (cv_set * Xstds[36]) + Xmeans[36]
diff = yy - cv_sety

for i in range(0,88):
    print(yy[i], "&", np.round(cv_sety[i],2), "&", np.round(diff[i],2), "&")

   ### =============== Measure all indices ================================
LOG_INFO(f"====> Overall R   = {pearsonr(yy,cv_sety)}", mcolor="red")
LOG_INFO(f"====> R^2 Score   = {metrics.r2_score(yy, cv_sety)}", mcolor="red")
LOG_INFO(f"====> MAE         = {metrics.mean_absolute_error(yy, cv_sety)}", mcolor="red")
LOG_INFO(f"====> MSE         = {metrics.mean_squared_error(yy, cv_sety)}", mcolor="red")
LOG_INFO(f"====> RMSE        = {RMSE(yy, cv_sety)}", mcolor="red")
LOG_INFO(f"====> MAPE        = {mean_absolute_percentage_error(yy, cv_sety)}", mcolor="red")


R_2 = metrics.r2_score(yy, cv_sety)
mae = metrics.mean_absolute_error(yy, cv_sety)

bland_altman_plot_paper(yy, cv_sety, "A_F_DNN_Hb_B")
act_pred_plot_paper(yy, cv_sety,R_2,mae,"A_F_DNN_Hb_R2")


# saved_m = tf.keras.models.load_model(keras_model_path) #loading model in saved_model format
# saved_m.summary()

# X_in_select = X[:1,]
# pred = saved_m.predict(X_in_select) ### Prediction of Hb Level
# pred = (pred * Xstds[36]) + Xmeans[36]
# print("Predicted Hb (mmol/L): " + str(pred))


#LOG     :====> Overall R   = 0.927250627316835
#LOG     :====> R^2 Score   = 0.8543437633026636
#LOG     :====> MAE         = 0.38505303879392927
#LOG     :====> MSE         = 1.2776561377412137
#LOG     :====> RMSE        = 1.1303345247055023
#LOG     :====> MAPE        = 6.265155245537811

# import tensorflow as tf
# h5_model = tf.keras.models.load_model("DNN_model_Hb.h5") # loading model in h5 format
# h5_model.summary()
# X_in_select = X[:1,]
# pred = h5_model.predict(X_in_select) ### Prediction of Hb Level
# pred = (pred * Xstds[36]) + Xmeans[36]
# print("Predicted Hb (mmol/L): " + str(pred))


# # generate related variables
# from numpy import mean
# from numpy import std
# from numpy.random import randn
# from numpy.random import seed
# from matplotlib import pyplot
# from scipy.stats import pearsonr, spearmanr
# # seed random number generator
# seed(1)
# # prepare data
# data1 = Xorg[:,36]
# data2 = Xorg[:, 37]
# # summarize
# print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
# print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# # plot
# pyplot.scatter(data1, data2)
# pyplot.show()
# corr, _ = pearsonr(data2,data1)
# spearman_corr, _ = spearmanr(data1, data2)
# print('spearman_corr correlation: %.3f' % spearman_corr)

# import seaborn as sns
# ax = sns.scatterplot(data1, data2)
# ax.set_title("Negatively Correlated")
# ax.set_xlabel("Hemoglobin (g/dL)");
# ax.set_ylabel("Glucose (mmol/L)");
for i in range(2,3):
    for j in range(0,36):
        print(np.round(Xorg[i, j],2), "&", end = " ")
