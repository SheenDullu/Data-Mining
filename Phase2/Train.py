import pickle

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def handle_null(data):
    data_na = data.dropna(thresh=22)
    data_na = data_na.interpolate('spline', order=3, axis=1)
    data_na = data_na.fillna(method='bfill', axis=1)
    data_na = data_na.fillna(method='ffill', axis=1)
    return data_na


classifiers = [
    RandomForestClassifier(n_estimators=30,random_state=777),
    GradientBoostingClassifier(random_state=777),
    SVC(gamma='auto',random_state=777),
    MLPClassifier(alpha=1, max_iter=1000,random_state=999),
    ]


cls_names = ["Random_Forest",
             "Gradient_Boosting",
             "SVM_RBF",
             "ML_Perceptron"
             ]


def sliding_window(series, func, window, step):
    return series.rolling(window).apply(func).dropna()[::step]


def get_poly_coeff(y_axis, order=3):
    x = np.array(y_axis.index)
    coef = np.polyfit(x, y_axis, order)
    return coef


def extract_sliding_stats_features(y_axis_df, window_size, window_step):

    features_mean = y_axis_df.apply(sliding_window, axis=1, args=(np.mean, 8, 6))
    features_mean = pd.DataFrame(features_mean).reset_index(drop=True)
    features_mean.columns = ['Mean_' + str(x) for x in range(1, features_mean.shape[1] + 1)]

    features_auc = y_axis_df.apply(sum, axis=1)
    features_auc = pd.DataFrame(features_auc).reset_index(drop=True)
    features_auc.columns = ['Auc_' + str(x) for x in range(1, 2)]

    features_skew = y_axis_df.apply(sliding_window, axis=1, args=(skew, 8, 6))
    features_skew = pd.DataFrame(features_skew).reset_index(drop=True)
    features_skew.columns = ['Skew_' + str(x) for x in range(1, features_mean.shape[1] + 1)]

    features_std = y_axis_df.apply(sliding_window, axis=1, args=(np.std, 8, 6))
    features_std = pd.DataFrame(features_std).reset_index(drop=True)
    features_std.columns = ['Std_' + str(x) for x in range(1, features_mean.shape[1] + 1)]

    return pd.concat([features_mean, features_std, features_skew, features_auc], axis=1)


def extract_poly_coefficient_features(y_df):
    poly_coefficient_features = y_df.apply(get_poly_coeff, axis=1, result_type='expand')
    poly_coefficient_features = pd.DataFrame(poly_coefficient_features).reset_index(drop=True)
    poly_coefficient_features.columns = ['Coeff_' + str(x) for x in range(1, poly_coefficient_features.shape[1] + 1)]
    return poly_coefficient_features


def extract_fft_features(y_axis_df):
    features_fft = abs(np.fft.fft(y_axis_df, axis=1))[:, 1:5]
    features_fft = pd.DataFrame(features_fft).reset_index(drop=True)
    features_fft.columns = ['FFT_' + str(x) for x in range(1, features_fft.shape[1] + 1)]
    return features_fft


def get_zero_crossings_values(row, no_of_zero_crossings=2):
    values = list(np.where(np.diff(np.sign(row)))[0][0:no_of_zero_crossings])
    if (len(values) < no_of_zero_crossings):
        for j in range(len(values), no_of_zero_crossings):
            values.append(-1)
    return values


def extract_cgm_vel_features(y_df, window_size, window_step):
    y_vel_df = y_df.diff(axis=1, periods=1).iloc[:, 1:]
    cgm_mean_velocity = y_vel_df.apply(sliding_window, axis=1, args=(np.mean, window_size, window_step))
    cgm_mean_velocity_features = pd.DataFrame(cgm_mean_velocity).reset_index(drop=True)
    cgm_mean_velocity_features.columns = ['Mean_CGM_Vel_' + str(x)
                                          for x in range(1, cgm_mean_velocity_features.shape[1] + 1)]

    cgm_std_velocity = y_vel_df.apply(sliding_window, axis=1, args=(np.std, window_size, window_step))
    cgm_std_velocity_features = pd.DataFrame(cgm_std_velocity).reset_index(drop=True)
    cgm_std_velocity_features.columns = ['Std_CGM_Vel_' + str(x)
                                         for x in range(1, cgm_std_velocity_features.shape[1] + 1)]

    cgm_skew_velocity = y_vel_df.apply(sliding_window, axis=1, args=(skew, window_size, window_step))
    cgm_skew_velocity_features = pd.DataFrame(cgm_skew_velocity).reset_index(drop=True)
    cgm_skew_velocity_features.columns = ['Skew_CGM_Vel_' + str(x)
                                          for x in range(1, cgm_skew_velocity_features.shape[1] + 1)]

    cgm_max_velocity = y_vel_df.apply(np.max, axis=1)
    cgm_max_velocity_features = pd.DataFrame(cgm_max_velocity).reset_index(drop=True)
    cgm_max_velocity_features.columns = ['Max_CGM_Vel_' + str(x)
                                         for x in range(1, cgm_max_velocity_features.shape[1] + 1)]
    zero_crossings = y_vel_df.apply(get_zero_crossings_values, axis=1)
    zero_crossings = np.stack(zero_crossings)
    zero_crossing_features = pd.DataFrame(zero_crossings).reset_index(drop=True)
    zero_crossing_features.columns = ['ZCross_' + str(x)
                                      for x in range(1, zero_crossing_features.shape[1] + 1)]

    zero_crossing_score_list = []
    for zidx, idx in zip(zero_crossing_features['ZCross_1'], zero_crossing_features['ZCross_1'].index):
        if (zidx >= 0 and zidx + 1 <= y_df.shape[1]):
            zero_crossing_score_list.append(y_df.iloc[idx, zidx + 1] - y_df.iloc[idx, zidx])
        else:
            zero_crossing_score_list.append(0)

    zero_crossing_features['ZCross_Score'] = pd.Series(zero_crossing_score_list)

    min_cgm_velocity = y_vel_df.apply(np.min, axis=1)
    min_cgm_velocity_features = pd.DataFrame(min_cgm_velocity).reset_index(drop=True)
    min_cgm_velocity_features.columns = ['Min_CGM_Vel_' + str(x)
                                         for x in range(1, min_cgm_velocity_features.shape[1] + 1)]

    return pd.concat([cgm_mean_velocity_features, cgm_std_velocity_features, cgm_skew_velocity_features,
                      cgm_max_velocity_features, min_cgm_velocity_features, zero_crossing_features], axis=1)


def extract_features(df):
    poly_coefficient_features = extract_poly_coefficient_features(df)
    fft_features = extract_fft_features(df)
    sliding_stats_features = extract_sliding_stats_features(df, 6, 4)
    cgm_vel_features = extract_cgm_vel_features(df, 6, 4)

    features = pd.concat([
        sliding_stats_features,
        poly_coefficient_features,
        fft_features,
        cgm_vel_features
    ]
        , axis=1)
    return features


def extract_features_test(data):
    data_cleaned = handle_null(data)
    data_preprocessed = data_cleaned
    features = extract_features(data_preprocessed)
    return features


def train_model(rolling_range=9):
    # cleaning data
    cleaned_meal_dataset = handle_null(meal_dataset)
    cleaned_no_meal_dataset = handle_null(no_meal_dataset)
    print("Meal data after handling null values: ", cleaned_meal_dataset.shape)
    print("No meal data after handling null values: ", cleaned_no_meal_dataset.shape)
    preprocessed_meal_dataset = cleaned_meal_dataset
    preprocessed_no_meal_dataset = cleaned_no_meal_dataset

    meal_data_features = extract_features(preprocessed_meal_dataset)
    print(meal_data_features.shape)

    no_meal_data_features = extract_features(preprocessed_no_meal_dataset)
    print(no_meal_data_features.shape)

    X = np.concatenate([meal_data_features, no_meal_data_features])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    file_handler = open("models/scaler.pkl", "wb")
    pickle.dump(scaler, file_handler)

    y = np.append(np.ones(len(meal_data_features)), np.zeros(len(no_meal_data_features)))

    print("X:", X.shape, ", y:", y.shape)

    for name, clf in zip(cls_names, classifiers):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

        scores = cross_validate(clf, X, y, cv=skf,
                                scoring=('accuracy', 'f1', 'precision', 'recall'),
                                return_train_score=True)
        print("-----" + name + "-----")
        print("Accuracy:", scores['test_accuracy'].mean(), "F1:", scores['test_f1'].mean())
        print("Recall:", scores['test_recall'].mean(), "Precision:", scores['test_precision'].mean())

        clf.fit(X, y)
        file_handler = open("models/" + name + ".model", "wb")
        pickle.dump(clf, file_handler)

    return meal_data_features, no_meal_data_features, X, y, preprocessed_meal_dataset


if __name__ == "__main__":
    column_names = [x for x in range(0, 31)]
    meal_dataset_1 = pd.read_csv('MealNoMealData/mealData1.csv', names=column_names)
    meal_dataset_2 = pd.read_csv('MealNoMealData/mealData2.csv', names=column_names)
    meal_dataset_3 = pd.read_csv('MealNoMealData/mealData3.csv', names=column_names)
    meal_dataset_4 = pd.read_csv('MealNoMealData/mealData4.csv', names=column_names)
    meal_dataset_5 = pd.read_csv('MealNoMealData/mealData5.csv', names=column_names)

    no_meal_dataset_1 = pd.read_csv('MealNoMealData/Nomeal1.csv', names=column_names)
    no_meal_dataset_2 = pd.read_csv('MealNoMealData/Nomeal2.csv', names=column_names)
    no_meal_dataset_3 = pd.read_csv('MealNoMealData/Nomeal3.csv', names=column_names)
    no_meal_dataset_4 = pd.read_csv('MealNoMealData/Nomeal4.csv', names=column_names)
    no_meal_dataset_5 = pd.read_csv('MealNoMealData/Nomeal5.csv', names=column_names)

    meal_dataset = pd.concat([meal_dataset_1, meal_dataset_2, meal_dataset_3, meal_dataset_4, meal_dataset_5])
    meal_dataset = meal_dataset.iloc[:, ::-1]
    meal_dataset.columns = column_names
    print(meal_dataset.shape)

    no_meal_dataset = pd.concat([no_meal_dataset_1, no_meal_dataset_2, no_meal_dataset_3,
                                 no_meal_dataset_4, no_meal_dataset_5])
    no_meal_dataset = no_meal_dataset.iloc[:, ::-1]
    no_meal_dataset.columns = column_names
    print(no_meal_dataset.shape)

    train_model()
