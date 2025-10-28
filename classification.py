import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)
from scipy.stats import shapiro

def feature_matrix(df_power, df_plv):
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

def classification(df: pd.DataFrame, task: str = "A", verbose: bool = False):
    """
    Runs classification pipeline for different scenarios.

    Parameters
    ----------
    :df: pd.DataFrame
        Contains a row per patient/time -- with all features as columns. 
    :metadata: pd.DataFrame
        Contains the patientID, timepoint and group of the cells.
    :features: pd.DataFrame
        Dataframe containing all features as columns and all patients as rows. 
    :task: str
        Type of classification you want to be done.
        - A: predictive at t0, tries to predict group before treatment
        - B: treatment effects at t2, tries to predict group after treatment
        - AB: both of the above
        ???
        - C: treatment effects for responders, tries to predict dosage (zero or max)
        - D: treatment effects fpr non-responders, tries to predict dosage (zero or max)
    :verbose: bool
        Option to be updated throughout running the script. 
    """
    results = {}

    def run(df_features, label):
        if verbose: print(f"Running {label}")
    
        # --- Data split --- #
        X = df_features.drop(columns=["Patient", "Time", "Group"])
        y = df_features["Group"].replace({"Responder": 1, "Non-responder": 0})
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)

        # --- Adaptive feature scaling --- #
        normal, outliers = 0, 0
        scaler_count = {"standard": 0, "minmax": 0, "robust": 0}
        scalers = {}
        for column in X_train.columns:
            column_data = X_train[column].values

            # Check for normal distribution
            _, pvalue = shapiro(column_data)
            not_normal_dist = 1 if float(pvalue) <= 0.05 else 0

            # Check for outliers
            Q1 = np.percentile(column_data, 25)
            Q3 = np.percentile(column_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
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
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        for col, scaler in scalers.items():
            scaler.fit(X_train[[col]])
            X_train_scaled[col] = scaler.transform(X_train[[col]])
            X_test_scaled[col] = scaler.transform(X_test[[col]])
        
        if verbose: print(f"Scaling summary: {scaler_count}")

        # --- Feature Reduction --- #
        # Variance
        selection = VarianceThreshold(threshold = 0.01)
        X_train_var = selection.fit_transform(X_train_scaled)
        X_test_var = selection.transform(X_test_scaled)
        selected_features = X_train_scaled.columns[selection.get_support()]

        # Correlation
        X_train_corr = pd.DataFrame(X_train_var, columns=selected_features)
        corr_matrix = X_train_corr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        X_train_corr = X_train_corr.drop(columns=to_drop)
        X_test_corr = pd.DataFrame(X_test_var, columns=selected_features).drop(columns=to_drop)
        X_test_corr = X_test_corr[X_train_corr.columns]

        if verbose: 
            print("Remaing features are: ({len(X_train_corr.columns)}):")
            for col in X_train_corr.columns:
                print(f" - {col}")

        # Workaround with NaNs
        imputer = SimpleImputer(strategy="mean")
        X_train_corr = pd.DataFrame(imputer.fit_transform(X_train_corr), columns=X_train_corr.columns)
        X_test_corr = pd.DataFrame(imputer.transform(X_test_corr), columns=X_train_corr.columns)

        # PCA
        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_corr)
        X_test_pca = pca.transform(X_test_corr)

        if verbose: print(f"Features dropped from {df_features.shape[1]} to {X_train_pca.shape[1]} after PCA.")

        # --- Feature selection (RFE) --- #
        estimator = LogisticRegression(max_iter = 500, class_weight = "balanced")
        n_features_to_select = min(20, X_train_pca.shape[1])
        rfe = RFE(estimator, n_features_to_select = n_features_to_select)
        X_train_sel = rfe.fit_transform(X_train_pca, y_train)
        X_test_sel = rfe.transform(X_test_pca)

        if verbose: 
            print(f"RFE selected {X_train_sel.shape[1]} features.")

        # --- Classifier: cross validation --- #
        clsfs = [LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),KNeighborsClassifier(),GaussianNB(),
                 LogisticRegression(),SGDClassifier(),RandomForestClassifier(), svm.SVC()]
        clf_names = ["Linear Discriminant Analysis", "Quadratic Discriminant Analysis", "K-Neighbors", "Gaussian",
                    "Logistic Regression", "SGD", "Random Forest", "SVC"]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = []

        print("\n--- Classifier baseline performance (CV AUC) ---")
        for name, clf in zip(clf_names, clsfs):
            aucs = cross_val_score(clf, X_train_sel, y_train, cv=cv, scoring='roc_auc')
            cv_results.append({
                "Model": name,
                "Mean AUC": np.mean(aucs),
                "Std AUC": np.std(aucs)
            })
            print(f"{name}: Mean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

        # --- Final classifier: SVC --- #
        svc = svm.SVC(probability=True)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'degree': [2, 3, 4]}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(
            svc,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            verbose=2)

        grid.fit(X_train_sel, y_train)

        print("Best parameters:", grid.best_params_)
        print(f"Best CV AUC: {grid.best_score_:.3f}")

        # --- Evaluation on test set --- #
        best_svc = grid.best_estimator_
        y_pred = best_svc.predict(X_test_sel)
        y_proba = best_svc.predict_proba(X_test_sel)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        print(f"Test ROC AUC: {auc:.3f}")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


    # --- Run task on label
    if task.lower() in ["a", "ab"]:
        run(df[df["Time"] == "t0"], label="Task A (Predictive, t0)")
    if task.lower() in ["b", "ab"]:
        run(df[df["Time"] == "t2"], label="Task B (Treatment, t2)")

    plt.show()
    return
