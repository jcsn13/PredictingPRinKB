from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_roc_curve, accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, \
    recall_score, precision_score
from GwasFeatureExtractor import GwasFeatureExtractor
import joblib as jb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import sys
import pandas as pd
import scipy.stats
import warnings

warnings.filterwarnings('ignore')

# ################################################## #
# Author: Jose Claudio Soares                        #
# Email: joseclaudiosoaresneto@gmail.com             #
# Date(mm/dd/yyyy): 11/03/2020                       #
# Github: jcsn13                                     #
# Credits: github.com/crowegian/AMR_ML               #
# ################################################## #


class PredictPRinKP:
    def __init__(self, datasets_full=None, datasets_gwas=None):
        self.dict_dataframes = {}
        self.dataset_full = datasets_full
        self.datasets_gwas = datasets_gwas
        self.SEED = 42
        self.models = {}

    def read_datasets(self):
        if self.dataset_full:
            for i in self.dataset_full:
                df_full = pd.read_csv('../data/' + i + 'full.csv')
                df_full.set_index('isolate', drop=True, inplace=True)

                pb_res = df_full['pbr_res']
                df_full = df_full.drop(columns=['pbr_res'], axis=1)

                self.dict_dataframes['pbr_res_' + i] = pb_res
                self.dict_dataframes['df_full_' + i] = df_full

        if self.datasets_gwas:
            for i in self.datasets_gwas:
                df_train = pd.read_csv('../data/' + i + 'train.csv')

                df_test = pd.read_csv('../data/' + i + 'test.csv')

                df_gwas = pd.read_csv('../data/' + i + 'gwas.csv')
                df_gwas.set_index('LOCUS_TAG')

                df_full = pd.concat([df_test, df_train])

                pb_res = df_full[['pbr_res']]

                df_full = df_full.drop(columns=['pbr_res'], axis=1)

                self.dict_dataframes['pbr_res_' + i] = pb_res
                self.dict_dataframes['df_full_' + i] = df_full
                self.dict_dataframes['df_gwas_' + i] = df_gwas

    def mean_confidence_interval(self, grid, x_test, y_test, xgboost, confidence=0.95, scoring='accuracy'):
        score_array = []
        for i in range(100):
            x_sample_array = []
            y_sample_array = []

            range_ = int(len(y_test) * 0.4)

            for j in range(range_):
                index = np.random.randint(0, len(x_test) - 1)
                x = x_test[index]
                y = y_test[index]
                x_sample_array.append(x)
                y_sample_array.append(y)

            if xgboost is True and scoring == 'roc_auc':
                y_pred = grid.predict_proba(np.array(x_sample_array))[:, 1]  # xgb input cannot be a list, must be a matrix
            elif scoring == 'roc_auc':
                y_pred = grid.predict_proba(x_sample_array)[:, 1]
            elif xgboost is True:
                y_pred = grid.predict(np.array(x_sample_array))  # xgb input cannot be a list, must be a matrix
            else:
                y_pred = grid.predict(x_sample_array)

            score = self.calculate_metrics(y_pred, y_sample_array, scoring)
            score_array.append(score)

        a = 1.0 * np.array(score_array)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, (m-h, m+h)

    @staticmethod
    def calculate_metrics(x, y, scoring):
        if scoring == 'accuracy':
            return accuracy_score(x, y)
        elif scoring == 'roc_auc':
            try:  # in case something goes wrong with the testing dataset because is too small roc_auc
                return roc_auc_score(y, x)
            except:
                print('error roc_auc')
                return 1
        elif scoring == 'balanced_accuracy':
            return balanced_accuracy_score(x, y)
        elif scoring == 'f1':
            return f1_score(x, y)
        elif scoring == 'recall':
            return recall_score(x, y, zero_division=1)
        else:
            return precision_score(x, y, zero_division=1)

    def print_performance_metrics(self, grid, name, x_test, y_test, dataset, short_name, xgboost=False):
        print("\n*********{}*********".format(name))
        print(
            "Performed GridSearchCV 10\nBest CV params {}\nBest CV auc_roc: {}"
                .format(grid.best_params_, grid.best_score_))

        metrics = ['roc_auc', 'balanced_accuracy', 'accuracy', 'f1', 'recall', 'precision']
        print("Best estimator metrics:")
        for metric in metrics:
            print("--> Test {}(confidence interval 0.95):{}"
                  .format(metric, self.mean_confidence_interval(grid.best_estimator_, x_test, y_test, scoring=metric,
                                                                xgboost=xgboost)))

        curve_name = short_name + " " + dataset
        plot_roc_curve(grid.best_estimator_, x_test, y_test, name=curve_name)
        plt.savefig('../plots/' + curve_name)
        # plt.show()

    def perform_train_full(self):
        print('\n############################# Full datasets #############################')
        for i in self.dataset_full:
            print('----------------Training {}----------------'.format(i))
            dataset = [self.dict_dataframes['df_full_' + i]]
            metadata = [self.dict_dataframes['pbr_res_' + i]]
            name = i

            self.perform_feature_selection(dataset, metadata, dataset_name=name)

            self.train_SGDClassifier(dataset, metadata, name)
            self.train_logistic_regression(dataset, metadata, name)
            self.train_SVC(dataset, metadata, name)
            self.train_random_forest(dataset, metadata, name)
            self.train_GBTC(dataset, metadata, name)
            self.perform_XGB(dataset, metadata, name)

            for model in self.models:
                self.save_trained_models(i, self.models[model], model)

            self.models = {}

        return self

    def perform_train_gwas(self):
        print('\n############################# GWAS datasets #############################')
        for i in self.datasets_gwas:
            print('----------------Training {}----------------'.format(i))
            dataset = [self.dict_dataframes['df_full_' + i],
                       self.dict_dataframes['df_gwas_' + i]]
            metadata = [self.dict_dataframes['pbr_res_' + i]]
            name = i

            self.perform_feature_selection(dataset, metadata, gwas=True, dataset_name=name)

            self.train_SGDClassifier(dataset, metadata, name)
            self.train_logistic_regression(dataset, metadata, name)
            self.train_SVC(dataset, metadata, name)
            self.train_random_forest(dataset, metadata, name)
            self.train_GBTC(dataset, metadata, name)
            self.perform_XGB(dataset, metadata, name)

            for model in self.models:
                self.save_trained_models(i, self.models[model], model)

            self.models = {}

        return self

    def perform_feature_selection(self, dataset, metadata, dataset_name, gwas=False):
        if gwas is True:
            gwas_feature_extractor = GwasFeatureExtractor(x_locus_tags=dataset[1]['LOCUS_TAG'], gwas_df=dataset[1])
            gwas_feature_extractor.fit()
            dataset[0] = gwas_feature_extractor.transform(dataset[0])

        encoder = LabelEncoder()
        isolate_column = dataset[0].columns[0]
        dataset[0][isolate_column] = encoder.fit_transform(dataset[0][isolate_column])
        n_features = 200

        estimator = SVC(kernel='linear', random_state=self.SEED)
        selector = RFECV(estimator=estimator, cv=10, step=1, min_features_to_select=n_features, scoring='roc_auc')
        selector = selector.fit(dataset[0], metadata[0].values.ravel())
        selected_features = dataset[0].columns[selector.support_]
        dataset[0] = dataset[0].filter(selected_features)
        
        print('Number of features selected:', len(selected_features))

        features_df = pd.DataFrame(selected_features, columns=['Features'])
        features_df.to_csv('../selected_features/'+dataset_name+'.csv', index=True)

        return self

    def train_SGDClassifier(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = train_test_split(dataset[0].values, metadata[0].values.ravel(),
                                                            test_size=0.25, random_state=self.SEED,
                                                            stratify=metadata[0].values.ravel())

        parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10], 'loss': ['log'],
                      'penalty': ['l1', 'l2']}
        model = SGDClassifier(random_state=self.SEED, max_iter=10000)
        grid = GridSearchCV(model, parameters, cv=10, return_train_score=True, scoring='roc_auc')
        grid.fit(x_train, y_train)

        model_info = ["Stochastic Gradient Descent Classifier", x_test, y_test, "SGDC"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2], name, model_info[3])

        self.models[model_info[-1]] = grid.best_estimator_

        return self

    def train_logistic_regression(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = train_test_split(dataset[0].values, metadata[0].values.ravel(),
                                                            test_size=0.25, random_state=self.SEED,
                                                            stratify=metadata[0].values.ravel())

        parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l2']}
        model = LogisticRegression(random_state=self.SEED, max_iter=10000)
        grid = GridSearchCV(model, parameters, cv=10, return_train_score=True, scoring='roc_auc')
        grid.fit(x_train, y_train)

        model_info = ["Logistic Regression", x_test, y_test, "LR"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2], name, model_info[3])

        self.models[model_info[-1]] = grid.best_estimator_

        return self

    def train_SVC(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = train_test_split(dataset[0].values, metadata[0].values.ravel(),
                                                            test_size=0.25, random_state=self.SEED,
                                                            stratify=metadata[0].values.ravel())
        parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10],
                      'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
        model = SVC(random_state=self.SEED, max_iter=10000, probability=True)
        grid = GridSearchCV(model, parameters, cv=10, return_train_score=True, scoring='roc_auc')
        grid.fit(x_train, y_train)

        model_info = ["Support Vectors Classifier", x_test, y_test, "SVC"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2], name, model_info[3])

        self.models[model_info[-1]] = grid.best_estimator_

        return self

    def train_GBTC(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = train_test_split(dataset[0].values, metadata[0].values.ravel(),
                                                            test_size=0.25, random_state=self.SEED,
                                                            stratify=metadata[0].values.ravel())

        parameters = {'learning_rate': [0.001, 0.01, 0.1], 'n_estimators': [50, 100, 200, 300, 400, 500],
                      'max_depth': [5, 10, 12, 15, None], 'min_samples_leaf': [20, 50, 100, 150],
                      'subsample': [0.6, 0.8, 1.0]}
        model = GradientBoostingClassifier(random_state=self.SEED)
        grid = GridSearchCV(model, parameters, cv=10, return_train_score=True, scoring='roc_auc')
        grid.fit(x_train, y_train)

        model_info = ["Gradient Boosting Classifier", x_test, y_test, "GBTC"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2], name, model_info[3])

        self.models[model_info[-1]] = grid.best_estimator_

        return self

    def train_random_forest(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = train_test_split(dataset[0].values, metadata[0].values.ravel(),
                                                            test_size=0.25, random_state=self.SEED,
                                                            stratify=metadata[0].values.ravel())

        parameters = {'n_estimators': [50, 100, 200, 400, 500], 'max_features': ['sqrt', 'log2'],
                      'max_depth': [10, 12, 15, 20, None], 'criterion': ['gini', 'entropy']}
        model = RandomForestClassifier(random_state=self.SEED)
        grid = GridSearchCV(model, parameters, cv=10, return_train_score=True, scoring='roc_auc')
        grid.fit(x_train, y_train)

        model_info = ["Random Forest", x_test, y_test, "RF"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2], name, model_info[3])

        self.models[model_info[-1]] = grid.best_estimator_

        return self

    def perform_XGB(self, dataset, metadata, name):
        x_train, x_test, y_train, y_test = train_test_split(dataset[0].values, metadata[0].values.ravel(),
                                                            test_size=0.25, random_state=self.SEED,
                                                            stratify=metadata[0].values.ravel())

        parameters = {'min_child_weight': [1, 5, 10], 'gamma': [0.5, 1, 1.5, 2, 5], 'subsample': [0.6, 0.8, 1.0],
                      'colsample_bytree': [0.6, 0.8, 1.0], 'max_depth': [3, 5, 10, 15, None],
                      'learning_rate': [0.01, 0.1, 0.2]}

        model = xgb.XGBClassifier(random_state=self.SEED, n_estimators=600,
                                  objective='binary:logistic', verbosity=0)
        grid = GridSearchCV(model, parameters, cv=10, return_train_score=True, scoring='roc_auc')
        grid.fit(x_train, y_train)

        model_info = ["XGBoost", x_test, y_test, "XGB"]
        self.print_performance_metrics(grid, model_info[0], model_info[1], model_info[2], name, model_info[3],
                                       xgboost=True)

        self.models[model_info[-1]] = grid.best_estimator_

        return self

    def save_trained_models(self, dataset, model, model_name):
        jb.dump(model, '../models/{}/{}.pkl'.format(dataset, dataset+model_name))

    def main(self):
        self.read_datasets()

        # if self.dataset_full:
        #     self.perform_train_full()
        if self.datasets_gwas:
            self.perform_train_gwas()


if __name__ == "__main__":
    # Uncomment to genetare log.txt
    # stdoutOrigin = sys.stdout
    # sys.stdout = open("../logs/log_fs_SVN_50estimators.txt", "w")

    df01 = ['dataset_1_','dataset_2_', 'dataset_3_', 'dataset_7_']
    df02 = ['dataset_19_', 'dataset_20_', 'dataset_21_']

    PR = PredictPRinKP(datasets_full=df01, datasets_gwas=df02)

    PR.main()

    # sys.stdout.close()
    # sys.stdout = stdoutOrigin

    print("finished model training")
