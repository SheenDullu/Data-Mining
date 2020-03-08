import math
import pandas as pds
import numpy as nump
import datetime as dt
import matplotlib.pyplot as pplot

from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# for polynomial coefficient values
def polynomial_coefficient(y_value, order=3):
    x_value = nump.array(y_value.index)
    return nump.polyfit(x_value, y_value, order)

# for slider window
def slider_window(series, funct, windows, step):
    return series.rolling(windows).apply(funct).dropna()[::step]

# mean for maximum excursion mean and standard deviation
def MaxE_slider_window_mean(y):
    maximum = y.idxmax()
    minimum = y.loc[0:maximum].idxmin()
    mean = y.loc[minimum:maximum].mean()
    return mean


def MaxE_slider_window_std(y):
    maximum = y.idxmax()
    minimum = y.loc[0:maximum].idxmin()
    st_deviation = y.loc[minimum:maximum].std(ddof=0)
    return st_deviation


def convert_matlab_to_datetime(matlab_to_datetime):
    if not (math.isnan(matlab_to_datetime)):
        day = dt.datetime.fromordinal(int(matlab_to_datetime))
        day_fraction = dt.timedelta(days=matlab_to_datetime % 1) - dt.timedelta(days=366)
        return day + day_fraction


# For loading data 
dataset_1 = pds.read_csv('DataFolder/CGMSeriesLunchPat2.csv')
time_set_1 = pds.read_csv('DataFolder/CGMDatenumLunchPat2.csv')
time_set_1 = time_set_1.applymap(convert_matlab_to_datetime)

# to preprocess and clean data
y_axis_df = dataset_1.dropna(thresh=24)
x_axis_df = time_set_1
x_axis_df = x_axis_df.loc[y_axis_df.index, :]
x_axis_df.columns = [k for k in range(1, x_axis_df.shape[1] + 1)]
y_axis_df.columns = [k for k in range(1, y_axis_df.shape[1] + 1)]
y_axis_df = y_axis_df.interpolate('polynomial', order=5, axis=1)
y_axis_df = y_axis_df.fillna(method='bfill', axis=1)
y_axis_df = y_axis_df.fillna(method='ffill', axis=1)
y_axis_df = y_axis_df.iloc[:, :-1]
x_axis_df = x_axis_df.iloc[:, :-1]
y_axis_df.shape
x_axis_df.shape


#to build cgm velocity features
y_velocity_df = y_axis_df.iloc[:, ::-1].diff(axis=1, periods=1).reset_index(drop=True)
cgm_mean_velocity = y_velocity_df.iloc[:, ::-1].apply(slider_window, axis=1, args=(nump.mean, 8, 6)).reset_index(drop=True) #
cgm_std_velocity = y_velocity_df.iloc[:, ::-1].apply(slider_window, axis=1, args=(nump.std, 8, 6)).reset_index(drop=True)  #

#For maximum excursion features
MaxE_mean = y_axis_df.apply(MaxE_slider_window_mean,axis=1).reset_index(drop=True)
MaxE_std = y_axis_df.apply(MaxE_slider_window_std,axis=1).reset_index(drop=True)


#for statistical features
features_mean = y_axis_df.apply(slider_window, axis=1, args=(nump.mean, 8, 6))  # .shape
features_mean = pds.DataFrame(features_mean).reset_index(drop=True)
features_mean.columns = ['Mean_' + str(k) for k in range(1, features_mean.shape[1] + 1)]

features_auc = y_axis_df.apply(sum, axis=1)
features_auc = pds.DataFrame(features_auc).reset_index(drop=True)
features_auc.columns = ['Auc_' + str(k) for k in range(1, 2)]

features_skew = y_axis_df.apply(slider_window, axis=1, args=(skew, 8, 6))
features_skew = pds.DataFrame(features_skew).reset_index(drop=True)
features_skew.columns = ['Skew_' + str(k) for k in range(1, features_mean.shape[1] + 1)]

features_std = y_axis_df.apply(slider_window, axis=1, args=(nump.std, 8, 6))
features_std = pds.DataFrame(features_std).reset_index(drop=True)
features_std.columns = ['StD_' + str(k) for k in range(1, features_mean.shape[1] + 1)]

#for FFT features
features_fft = abs(nump.fft.fft(y_axis_df, axis=1))[:, 1:5]
features_fft = pds.DataFrame(features_fft).reset_index(drop=True)
features_fft.columns = ['FFT_' + str(k) for k in range(1, features_mean.shape[1] + 1)]
features_fft.shape

#for polynomial curve fitting features
poly_coefficient_features = y_axis_df.apply(polynomial_coefficient, axis=1, result_type='expand')
poly_coefficient_features = pds.DataFrame(poly_coefficient_features).reset_index(drop=True)
poly_coefficient_features.columns = ['Coef_' + str(k) for k in range(1, features_mean.shape[1] + 1)]
poly_coefficient_features.shape


#Feature matrix creation
feature_matrix = pds.concat([features_mean, features_std, features_skew, features_auc,
                      features_fft, poly_coefficient_features, cgm_mean_velocity, cgm_std_velocity, MaxE_mean, MaxE_std], axis=1)
feature_matrix.shape

#Scaling the feature
scaled_feature_matrix = StandardScaler().fit_transform(feature_matrix.values)
scaled_feature_matrix.shape


#PCA component
pca = PCA(5)
pca.fit(scaled_feature_matrix)

print(scaled_feature_matrix.shape)


#to compute the features in the reduced dimensions
reduced_features_pca = pca.transform(scaled_feature_matrix)
print(reduced_features_pca.shape)


scaled_feature_df = pds.DataFrame(scaled_feature_matrix)
for j in range(scaled_feature_df.shape[1]):
    pplot.scatter(x=scaled_feature_df.index, y=scaled_feature_df.iloc[:, j])

pds.DataFrame(pca.components_).T
plot_df = pds.DataFrame(pca.components_)
plot_df.columns = feature_matrix.columns
plot_df.T.plot(kind='bar', figsize=(15, 8))
pplot.show()
