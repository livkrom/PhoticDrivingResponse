"""
Module classification chooses the best classifier and performs the classification.
"""
from dataclasses import dataclass, field
from typing import Dict, Any
import warnings
import numpy as np
import pandas as pd

from sklearn.exceptions import UndefinedMetricWarning

from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report)
from sklearn.exceptions import ConvergenceWarning

from scipy.stats import shapiro
warnings.filterwarnings(
    "ignore",
    message=".*covariance matrix of class.*not full rank.*",
    category=RuntimeWarning,
    module="sklearn.discriminant_analysis")
warnings.filterwarnings(
    "ignore",
    message=".*covariance matrix of class.*not full rank.*",
    category=UserWarning,
    module="sklearn.discriminant_analysis")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

@dataclass
class Classifier:
    """
    Chooses a classifier based on a 10-time repeated cross-validation of claasifiers.
    Performs classification.
    """
    df_power: pd.DataFrame
    df_plv: pd.DataFrame
    n_repeats: int = 20
    results: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    def run(self, task: str = "A"):
        """
        Runs the module. 
        """
        task = task.lower()
        comparisons = []

        # Define comparisons depending on the task
        if "a" in task:
            comparisons.append({"type": "time", "value": "t0"})
        if "b" in task:
            comparisons.append({"type": "time", "value": "t2"})
        if "c" in task:
            comparisons.append({"type": "group", "value": "Responder"})
        if "d" in task:
            comparisons.append({"type": "group", "value": "Non-responder"})
        if not comparisons:
            raise ValueError(f"Unknown task '{task}'. Use A, B, C, or D.")

        for comp in comparisons:
            df_features = self._feature_matrix(self.df_power, self.df_plv)

            if comp["type"] == "time":
                tp = comp["value"]
                print(f"\nRunning classification at timepoint {tp} (Responder vs Non-responder)")
                df_task = df_features[df_features["Time"] == tp]
                target = "group"
            else:
                group = comp["value"]
                print(f"\nRunning classification in group {group} (T0 vs T2)")
                df_task = df_features[df_features["Group"] == group]
                target = "time"

            auc_tracking = []
            all_cv_results = []
            feature_records = []
            transformation_records = []

            for i in range(self.n_repeats):
                print(f"--- Run {i+1}/{self.n_repeats}")

                x_train_sel, x_test_sel, y_train, y_test, t_record = self._feature_selection(df_task, target=target)
                transformation_records.append(t_record)
                feature_records.append(t_record["rfe_features"])

                cv_results = self._cross_val_classifiers(x_train_sel, y_train)
                all_cv_results.append(pd.DataFrame(cv_results))

                for row in cv_results:
                    auc_tracking.append({"Iteration": i,
                                         "Model": row["Model"],
                                         "Mean AUC": row["Mean AUC"]})

            combined = pd.concat(all_cv_results)
            summary = combined.groupby("Model").agg({"Mean AUC": ["mean", "std"]}).reset_index()
            summary.columns = ["Model", "Mean AUC (mean)", "Mean AUC (std)"]
            self.results = summary.sort_values("Mean AUC (mean)", ascending=False)

            print("\n=== Summary over all repetitions ===")
            print(summary.sort_values("Mean AUC (mean)", ascending=False))

            best_model_name = self.results.iloc[0]["Model"]
            print(f"Choosing classifier {best_model_name} as final classifier.")

            auc_df = pd.DataFrame(auc_tracking)
            best_iter_row = auc_df[auc_df["Model"] == best_model_name].sort_values("Mean AUC", ascending=False).iloc[0]
            best_iter_idx = int(best_iter_row["Iteration"])

            best_transform_record = transformation_records[best_iter_idx]
            df_task_final = df_task.copy()

            if comp["type"] == "time":
                df_task = df_features[df_features["Time"] == tp]
                x = df_task_final.drop(columns=["Patient", "Time", "Group"])
                y = df_task_final["Group"].map({"Responder": 1, "Non-responder": 0}).astype(int)

            else:
                df_task = df_features[df_features["Group"] == group]
                x = df_task_final.drop(columns=["Patient", "Time", "Group"])
                y = df_task_final["Time"].map({"t0": 0, "t2": 1}).astype(int)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y)
            x_train_sel, x_test_sel = self._apply_transformations_from_record(best_transform_record, x_train, x_test)
            self._classifier_final(x_train_sel, x_test_sel, y_train, y_test, best_model_name)


    def _feature_matrix(self, df_power, df_plv):
        """
        Combines the feature dataframe to one dataframe useable for classification. 

        Paramaters
        ----------
        :df_power: pd.DataFrame
            Dataframe with power values data.
        :df_plv: pd.DataFrame
            Datafrane with plv values data. 

        Returns
        -------
        :df_features: pd.DataFrame
            Contains a row per patient/time -- with all features as columns. 
        :metadata: pd.DataFrame
            Contains the patientID, timepoint and group of the cells.
        :features: pd.DataFrame
            Dataframe containing all features as columns and all patients as rows. 
        """
        cols_power = ["Patient", "Time", "Group", "FreqPairCSV", "Average_SNR", "Average_PWR", "Average_BASE"]
        cols_plv = ["Patient", "Time", "Group", "FreqPairCSV", "PLV"]

        df_plv = df_plv[df_plv["Time"].str.lower() != "base"]
        df_merged = pd.merge(df_power[cols_power], df_plv[cols_plv], how = "outer",
                            on = ["Patient", "Time", "Group", "FreqPairCSV"])
        df_merged = df_merged.rename(columns = {"Average_PWR": "Power_Abs",
                                            "Average_SNR": "Power_SNR", 
                                            "Average_BASE": "Power_Base"})

        df_wide = df_merged.melt(
            id_vars = ["Patient", "Time", "Group", "FreqPairCSV"],
            var_name = "Metric", value_name = "Value")
        df_wide["FeatureName"] =  df_wide["Metric"] + "__" + df_wide["FreqPairCSV"]
        df_features = df_wide.pivot_table(index=["Patient", "Time", "Group"],
                                        columns = "FeatureName", values = "Value").reset_index()

        return df_features

    def _feature_selection(self, df_features, target: str = "group"):
        """
        Prepares the dataframe for classification process.
        
        Paramaters
        ----------
        :df_features: pd.DataFrame
            Contains a row per patient/time -- with all features as columns.

        Returns
        -------
        :X_train_scaled: 
        :X_test_scaled: 
        :y_train: 
        :y_test: 
        """
        x = df_features.drop(columns=["Patient", "Time", "Group"])
        if target == "group":
            y = df_features["Group"].map({"Responder": 1, "Non-responder": 0}).astype(int)
        elif target == "time":
            y = df_features["Time"].map({"t0": 0, "t2": 1}).astype(int)
        else:
            raise ValueError("target must be 'group' or 'time'")

        if len(np.unique(y)) < 2:
            raise ValueError(f"Not enough classes in target for target='{target}'. Unique labels: {np.unique(y)}")

        # --- Data split --- #
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y)

        # --- Adaptive feature scaling --- #
        outliers = 0
        scaler_count = {"standard": 0, "minmax": 0, "robust": 0}
        scalers = {}
        for column in x_train.columns:
            column_data = x_train[column].values

            # Check for normal distribution
            _, pvalue = shapiro(column_data)
            not_normal_dist = 1 if float(pvalue) <= 0.05 else 0

            # Check for outliers
            q1 = np.percentile(column_data, 25)
            q3 = np.percentile(column_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_count = np.sum((column_data < lower_bound) | (column_data > upper_bound))
            if outliers_count > 0:
                outliers += 1

            # Choose appropriate scaler for each feature
            if not_normal_dist == 0:
                scalers[column] = StandardScaler()
                scaler_count["standard"] += 1
            elif not_normal_dist != 0 and outliers_count == 0:
                scalers[column] = MinMaxScaler()
                scaler_count["minmax"] += 1
            else:
                scalers[column] = RobustScaler()
                scaler_count["robust"] += 1

        # Scaling
        x_train_scaled = x_train.copy()
        x_test_scaled = x_test.copy()
        for col, scaler in scalers.items():
            scaler.fit(x_train[[col]])
            x_train_scaled[col] = scaler.transform(x_train[[col]])
            x_test_scaled[col] = scaler.transform(x_test[[col]])

        # Variance
        variance_selector = VarianceThreshold(threshold = 0.01)
        x_train_var = variance_selector.fit_transform(x_train_scaled)
        x_test_var = variance_selector.transform(x_test_scaled)
        selected_features = x_train_scaled.columns[variance_selector.get_support()]

        # Correlation
        x_train_corr = pd.DataFrame(x_train_var, columns=selected_features)
        corr_matrix = x_train_corr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        x_train_corr = x_train_corr.drop(columns=to_drop)
        x_test_corr = pd.DataFrame(x_test_var, columns=selected_features).drop(columns=to_drop)
        x_test_corr = x_test_corr[x_train_corr.columns]

        # Workaround with NaNs
        imputer = SimpleImputer(strategy="mean")
        x_train_corr = pd.DataFrame(imputer.fit_transform(x_train_corr), columns=x_train_corr.columns)
        x_test_corr = pd.DataFrame(imputer.transform(x_test_corr), columns=x_train_corr.columns)

        # PCA
        pca = PCA(n_components=0.95)
        x_train_pca = pca.fit_transform(x_train_corr)
        x_test_pca = pca.transform(x_test_corr)

        # --- Feature selection (RFE) --- #
        estimator = LogisticRegression(max_iter = 500, class_weight = "balanced")
        n_features_to_select = min(20, x_train_pca.shape[1])
        rfe = RFE(estimator, n_features_to_select = n_features_to_select)
        x_train_sel = rfe.fit_transform(x_train_pca, y_train)
        x_test_sel = rfe.transform(x_test_pca)
        rfe_features = rfe.get_support(indices=True)

        t_record = dict(
            scalers=scalers,
            variance_selector=variance_selector,
            to_drop=to_drop,
            imputer=imputer,
            pca=pca,
            rfe=rfe,
            rfe_features=rfe_features,
            final_columns_after_var_corr=x_train_corr.columns
        )
        return x_train_sel, x_test_sel, y_train, y_test, t_record

    def _apply_transformations_from_record(self, t_record: Dict[str, Any], x_train, x_test):
        """
        Applies final feature selection. 
        """
        # SCALERS
        for col, scaler in t_record["scalers"].items():
            x_train.loc[:, col] = scaler.transform(x_train[[col]]).flatten()
            x_test.loc[:, col] = scaler.transform(x_test[[col]]).flatten()

        # VARIANCE
        x_train_var = t_record["variance_selector"].transform(x_train)
        x_test_var = t_record["variance_selector"].transform(x_test)
        selected_columns_var = x_train.columns[t_record["variance_selector"].get_support()]

        # CORRELATION
        x_train_corr = pd.DataFrame(x_train_var, columns=selected_columns_var).drop(columns=t_record["to_drop"])
        x_test_corr = pd.DataFrame(x_test_var, columns=selected_columns_var).drop(columns=t_record["to_drop"])
        x_test_corr = x_test_corr[x_train_corr.columns]

        # IMPUTER
        x_train_corr = pd.DataFrame(t_record["imputer"].transform(x_train_corr), columns=x_train_corr.columns)
        x_test_corr = pd.DataFrame(t_record["imputer"].transform(x_test_corr), columns=x_train_corr.columns)

        # PCA
        x_train_pca = t_record["pca"].transform(x_train_corr)
        x_test_pca = t_record["pca"].transform(x_test_corr)

        # RFE
        x_train_sel = t_record["rfe"].transform(x_train_pca)
        x_test_sel = t_record["rfe"].transform(x_test_pca)
        return x_train_sel, x_test_sel

    def _cross_val_classifiers(self, x_train_sel, y_train):
        """
        Performs a cross-validation for different classifiers, chooses final classifier.

        Paramaters
        ----------
        :X_train_scaled: 
        :y_train:
            

        Returns
        -------
        :cv_results: 
        """
        clsfs = [LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),KNeighborsClassifier(),GaussianNB(),
                    LogisticRegression(),SGDClassifier(),RandomForestClassifier(), svm.SVC()]
        clf_names = ["Linear Discriminant Analysis", "Quadratic Discriminant Analysis", "K-Neighbors", "Gaussian",
                    "Logistic Regression", "SGD", "Random Forest", "SVC"]
        cv = StratifiedKFold(n_splits=3, shuffle=True)
        cv_results = []

        for name, clf in zip(clf_names, clsfs):
            aucs = cross_val_score(clf, x_train_sel, y_train, cv=cv, scoring='roc_auc')
            cv_results.append({
                "Model": name,
                "Mean AUC": np.mean(aucs),
                "Std AUC": np.std(aucs)})

        return cv_results

    def _classifier_final(self, x_train_sel, x_test_sel, y_train, y_test, best_model_name):
        """
        Performs hyperparameter tuning and classification for the final classifier.
        """

        # Hyperparameters per classifier
        classifiers = {
        "Linear Discriminant Analysis": (LinearDiscriminantAnalysis(), None),
        "Quadratic Discriminant Analysis": (QuadraticDiscriminantAnalysis(), None),
        "K-Neighbors": (KNeighborsClassifier(), {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }),
        "Gaussian": (GaussianNB(), None),
        "Logistic Regression": (LogisticRegression(max_iter=500), {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs", "saga"]
        }),
        "SGD": (SGDClassifier(max_iter=1000, tol=1e-3), {
            "loss": ["hinge", "log", "modified_huber"],
            "alpha": [0.0001, 0.001, 0.01]
        }),
        "Random Forest": (RandomForestClassifier(), {
            "n_estimators": [10, 20, 30, 40, 50, 60],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }),
        "SVC": (svm.SVC(probability=True), {
            "C": [0.01, 0.1, 1, 10, 100],
            "kernel": ['linear', 'rbf', 'poly', 'sigmoid'],
            "gamma": ['scale', 'auto', 0.01, 0.1, 1],
            "degree": [2, 3, 4]
        })}

        clf, param_grid = classifiers[best_model_name]
        if param_grid is not None:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid = GridSearchCV(clf, param_grid=param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
            grid.fit(x_train_sel, y_train)
            best_model = grid.best_estimator_
            print(f"\nBest parameters for {best_model_name}: {grid.best_params_}")
            print(f"Best CV AUC: {grid.best_score_:.3f}")
        else:
            best_model = clf.fit(x_train_sel, y_train)
            print(f"No hyperparameter grid for {best_model_name}, fitted default parameters.")

        # Evaluate on test set
        y_pred = best_model.predict(x_test_sel)
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(x_test_sel)[:, 1]
        else:  # For classifiers without predict_proba
            y_proba = best_model.decision_function(x_test_sel)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # scale to 0-1

        auc = roc_auc_score(y_test, y_proba)
        print(f"Test ROC AUC: {auc:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
