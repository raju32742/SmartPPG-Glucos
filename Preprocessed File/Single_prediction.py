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
from scipy import stats

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Dropout
from keras import optimizers
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers 
import random
import pandas as pd
import numpy as np

# Import other files
from model import*
# from measrmnt_indices import* 



df = pd.read_csv("H:/Glucose/Preprocessed File/preprocessed_PPG-34.csv")
df.drop(df.columns[[0, 1, 40]], axis=1, inplace=True) 
df.shape
Xorg = df.to_numpy()
scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorg)
Xmeans = scaler.mean_
Xstds = scaler.scale_

Gl_means, GL_std = Xmeans[37], Xstds[37]
Hb_means, Hb_std = Xmeans[36], Xstds[36]


input_data= "H:/Glucose/dataset_folder/ppg_feats.csv"

dataFr = pd.read_csv(input_data)
dataFr.drop(dataFr.columns[[0, 1]], axis=1, inplace=True) 
dataFr.insert(0, 'Age', float(40))
dataFr.insert(1, 'Sex(M/F)', 1)
dataFr.head()
dataFr.shape

#Add two dataframe 
df_in = dataFr.append(df, ignore_index = True)

#standardization 
Xorg_in = df_in.to_numpy()
scaler_in = StandardScaler()
Xscaled_in = scaler_in.fit_transform(Xorg_in)
## store these off for predictions with unseen data
Xmeans_in = scaler_in.mean_
Xstds_in = scaler_in.scale_
X = Xscaled_in[:1, 0:36]


get_best_ind_final_Gl = [0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18, 20, 21, 22, 26, 27, 30, 32, 35] 
get_best_ind_final_Hb = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 14, 18, 19, 20, 21, 22, 24, 26, 27, 30, 32, 35] 
X_Gl = X[:, get_best_ind_final_Gl]
X_Hb = X[:, get_best_ind_final_Hb]


# keras_model_path_Gl = 'H:/Glucose/Preprocessed File/GL Model'
# saved_m = tf.keras.models.load_model(keras_model_path_Gl) #loading model in saved_model format
# # saved_m.summary()

model_Gl = tf.keras.models.load_model('DNN_model_Gl.h5')
Gl = model_Gl.predict(X_Gl) ### Estimated of Gl Level
G_estimate = (Gl * GL_std) + Gl_means
print("Estimated Gl (mmol/L): " + str(G_estimate))


# Gl = saved_m.predict(X_Gl) ### Estimated of Gl Level
# G_estimate = (Gl * GL_std) + Gl_means
# print("Estimated Gl (mmol/L): " + str(G_estimate))



# keras_model_path_Hb = 'H:/Glucose/Preprocessed File/Hb Model'
# saved_m = tf.keras.models.load_model(keras_model_path_Hb) #loading model in saved_model format
# # saved_m.summary()

model_Hb = tf.keras.models.load_model('DNN_model_Hb.h5')
Hb = model_Hb.predict(X_Hb) ### Estimated of Hb Level
Hb_estimate = (Hb * Hb_std) + Hb_means
print("Estimated Hemoglobin (g/dL): " + str(Hb_estimate))


# Hb = saved_m.predict(X_Hb) ### Estimated of Hb Level
# Hb_estimate = (Hb * Hb_std) + Hb_means
# print("Estimated Hemoglobin (g/dL): " + str(Hb_estimate))



