#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import sys
sys.path.append('../')
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import argparse
import pandas as pd
from numpy import array
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot
from GA.features_selection import Feature_Selection_GA_Filter, \
    Feature_Selection_GA_Wrap
from sklearn.svm import SVR
from GA.measurement_indices import pearsonr, rmse, \
    RMSE, mean_absolute_percentage_error, index_agreement
from GA.measurement_indices import bland_altman_plot, act_pred_plot, bland_altman_plot_paper, act_pred_plot_paper
from PPG.utils import LOG_INFO
from model.model import DNN, DNN_3Layers
from model.config import config
#### Seed
np.random.seed(config.SEED)


# def main(dataset_dir, operation_type, ga_method, label_name, save_dir, verbose ):
    ##---------------------------------
    # dataset_dir     =   args.dataset_dir
    # operation_type  =   args.operation_type
    # ga_method       =   args.ga_method
    # label_name      =   args.label_name
    # save_dir        =   args.save_dir
    # verbose         =   args.verbose 
    # ##---------------------------------

    ## Load dataset (.csv) file
df = pd.read_csv(dataset_dir)
df_Clinical = pd.read_csv(clinical_dir)
df_Clinical.drop(df_Clinical.columns[[1, 2]], axis=1, inplace=True)

df_merge = pd.merge(df, df_Clinical, on='ID', how='inner')
df_merge.shape

col_list = ['ID', 'Age', 'Sex(M/F)','Systolic_peak(x)', 'Max. Slope(c)', 'Time of Max. Slope(t_ms)',\
            'Prev. point a_ip(d)', 'Time of a_ip(t_ip)', 'Diastolic_peak(y)', 'Dicrotic_notch(z)', \
            'Pulse_interval(tpi)', 'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)', \
            'Dicrotic_notch_time(t3)', 'w', 'Inflection_point_area_ratio(A2/A1)', \
            'a1','b1', 'e1', 'l1', 'a2','b2','e2', 'ta1', 'tb1', 'te1', 'tl1', 'ta2',\
            'tb2', 'te2', 'Fundamental_component_frequency(fbase)', 'Fundamental_component_magnitude(|sbase|)',\
            '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)', '3rd_harmonic_frequency(f3rd)', \
            '3rd_harmonic_magnitude(|s3rd|)', 'Stress-induced_vascular_response_index(sVRI)', 'Hb (g/dL)',\
            'Gl (mmol/L)', 'Cr (ml/dl)']
print(len(col_list))

df_merge = df_merge[col_list]
dict = {'Sex(M/F)':{'M':1, 'F':0}}      # label = column name
df_merge.replace(dict,inplace = True) 
df_merge.to_csv("H:/non-invasive-BP-measurement-from-fingertip-video-main/dataset_folder/preprocessed_PPG-34.csv")




dataFr = pd.read_csv(final_data_dir)
dataFr.drop(dataFr.columns[[0, 1, 38, 40]], axis=1, inplace=True)
dataFr.head()
dataFr.shape

Xorg = dataFr.to_numpy()
scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorg)
Xmeans = scaler.mean_
Xstds = scaler.scale_

y = Xscaled[:, 36]
X = Xscaled[:, 0:36]


## Load DNN model
# model = DNN(Xscaled) 
model = DNN(X)
n_splits = config.N_SPLITS
# R_2 = []
# mae = []
# mse= []
# rmse = []
# mape= []
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

    # ----------------------------------------
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5) 

    model.fit(x_train,y_train, epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE,
                        shuffle=True,  callbacks=[callback],  validation_split=0, verbose=1)
    # ----------------------------------------
    predicted_y =  model.predict(x_test)
    # R_2.append(metrics.r2_score(y_test, predicted_y))
    # y_test = (y_test * Xstds[36]) + Xmeans[36]
    # predicted = (predicted_y * Xstds[36]) + Xmeans[36]
    # mae.append(metrics.mean_absolute_error(y_test, predicted))
    # mse.append(metrics.mean_squared_error(y_test,predicted))
    # rmse.append(RMSE(y_test, predicted))
    # mape.append(mean_absolute_percentage_error(y_test, predicted))
    LOG_INFO(f"Individual R = {pearsonr(y_test, predicted_y)}", mcolor="green")
    print("R: " + str(pearsonr(y_test, predicted_y)))
    cv_set[test_index] = predicted_y[:, 0]




# print("R^2 (Avg. +/- Std.) is  %0.3f +/- %0.3f" %(np.mean(R_2),np.std(R_2)))    
# print("MAE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mae),np.std(mae)))    
# print("MSE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mse),np.std(mse)))   
# print("RMSE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.sqrt(np.mean(mse)),np.sqrt(np.std(mse))))
# print("MAPE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mape),np.std(mape)))

## For Get real values of target label
yy = (y * Xstds[36]) + Xmeans[36]
cv_set = (cv_set * Xstds[36]) + Xmeans[36]

   ### =============== Measure all indices ================================
LOG_INFO(f"====> Overall R   = {pearsonr(yy,cv_set)}", mcolor="red")
LOG_INFO(f"====> R^2 Score   = {metrics.r2_score(yy, cv_set)}", mcolor="red")
LOG_INFO(f"====> MAE         = {metrics.mean_absolute_error(yy, cv_set)}", mcolor="red")
LOG_INFO(f"====> MSE         = {metrics.mean_squared_error(yy, cv_set)}", mcolor="red")
LOG_INFO(f"====> RMSE        = {RMSE(yy, cv_set)}", mcolor="red")
LOG_INFO(f"====> MAPE        = {mean_absolute_percentage_error(yy, cv_set)}", mcolor="red")

R_2 = metrics.r2_score(yy, cv_set)
mae = metrics.mean_absolute_error(yy, cv_set)

yy = [4.27 ,4.44 ,5.6 ,6.7 ,12.8 ,4.44 ,5.33 ,4.83 ,4.44 ,6.61 ,6.55 ,4.66 ,5.61 ,6.28 ,5.67 ,5.28 ,4.83 ,5.0 ,4.94 ,12.94 ,8.67 ,5.89 ,4.83 ,11.39 ,6.22 ,4.78 ,11.11 ,5.44 ,5.33 ,12.83 ,6.39 ,6.83 ,5.67 ,5.0 ,4.55 ,13.06 ,11.0 ,5.39 ,5.56 ,4.66 ,6.83 ,5.22 ,5.5 ,4.94 ,5.55 ,6.1 ,6.7 ,6.1 ,5.4 ,5.4 ,5.2 ,6.8 ,6.1 ,5.5 ,5.7 ,5.8 ,5.3 ,4.9 ,5.3 ,5.4 ,5.0 ,5.3 ,6.1 ,7.5 ,5.6 ,6.4 ,6.56 ,5.0 ,6.72 ,14.560000000000002 ,4.89 ,6.3 ,6.33 ,6.22 ,10.33 ,21.11 ,9.78 ,3.4 ,8.2 ,6.39 ,3.3299999999999996 ,3.94 ,5.0 ,3.72 ,4.83 ,5.8 ,10.2 ,16.33]
cv_set = [4.294113796854485 ,4.5513882063129225 ,5.338871936502269 ,6.96400126092234 ,12.057976107450841 ,4.521065135899281 ,5.449700886632307 ,4.700250120003731 ,4.541584406972264 ,6.478026339544458 ,6.7230477323500155 ,4.646223567406753 ,5.649237845409131 ,5.788990120261116 ,5.352620707186857 ,5.18869636081558 ,4.780115242992547 ,5.060360183331288 ,5.296191073311902 ,11.782237630875223 ,8.421356739051163 ,5.688007834192697 ,5.187257204378375 ,11.815796102086646 ,6.1837013125684175 ,4.809372540571399 ,9.751791416840382 ,5.59513681093131 ,5.411741696444594 ,12.209503604553772 ,6.3359084900453375 ,6.655093837430678 ,5.509244412220007 ,4.951935820058269 ,4.312134160971642 ,12.754491689160941 ,10.411145056747245 ,5.27507364454188 ,5.59986096371121 ,4.469317498591051 ,6.700063125412713 ,5.337273631503978 ,5.4747470796306885 ,4.8189115351276115 ,5.303206540266778 ,5.8837277958441625 ,6.853794646317486 ,6.200212732231599 ,5.502977129485531 ,5.4938502207694615 ,5.116090697149917 ,6.6669246111201 ,6.343581164392915 ,5.476829029755353 ,5.441298958506696 ,5.915328571483622 ,5.259200944646397 ,4.720454813620409 ,5.245422416634863 ,5.5664122226429065 ,4.630248133882522 ,5.325124582833251 ,6.193368104216254 ,7.646429538072086 ,5.453062313252252 ,6.333782324606748 ,6.494641777006748 ,5.064111200669219 ,5.697221269421945 ,12.615184888578442 ,4.691384916350549 ,6.372703801210981 ,6.357315144977501 ,6.103407232397841 ,10.135685023730922 ,22.95615428469594 ,8.933379373462511 ,2.8601556812643723 ,8.117285213301532 ,6.260586008998388 ,3.6189919619503494 ,3.7707424618784424 ,5.092906019791758 ,3.4912384435553108 ,4.919861495528364 ,5.8149232764421805 ,10.491591864627111 ,11.854276931130673]



# yy = np.array(yy).reshape(-1, 1)
# cv_set = np.array(cv_set).reshape(-1, 1)

yy = np.array(yy)
cv_set = np.array(cv_set)

scaler = StandardScaler()
yy = scaler.fit_transform(yy)
cv_set = scaler.fit_transform(cv_set)


bland_altman_plot_paper(yy, cv_set, "MIC_B")
act_pred_plot_paper(yy, cv_set,R_2,mae,"MIC_R2")







if __name__=="__main__":
    '''
        parsing and executions
    '''
    # parser = argparse.ArgumentParser("Model train With-GA and Without-GA Script")
    # parser.add_argument("dataset_dir", help="Path to .csv generated dataset") 
    # parser.add_argument("--operation_type",type=str,default="Without-GA",help ="Which operation? (With-GA or Without-GA)")
    # parser.add_argument("--ga_method", type=str,default="filter",help ="Which GA-method? (filter or wrap)")
    # parser.add_argument("--label_name",type=str,default="SYS BP",help ="Name of Target Label(SYS BP or DYS BP)")
    # parser.add_argument("--save_dir",type=str,default="../output_figs/",help ="Save Plot Figures Directory.")
    # parser.add_argument("--verbose",type=int,required=False,default=1,help ="Whether you will see message/figure in terminal: default=1")

    # args = parser.parse_args()
    # main(args)
    
    dataset_dir = "../dataset_folder/ppg_feats.csv"
    clinical_dir = "../dataset_folder/data.csv"
    final_data_dir= "../dataset_folder/preprocessed_PPG-34.csv"
    operation_type = "Without-GA"
    ga_method = "filter"
    label_name = "Gl (mmol/L)"
    save_dir = "../output_figs/"
    verbose = 1
    # main(dataset_dir, clinical_dir, operation_type, ga_method, label_name, save_dir, verbose)