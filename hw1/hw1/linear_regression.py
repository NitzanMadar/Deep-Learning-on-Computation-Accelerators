import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.dot(X, self.weights_)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution you derived.
        #  Use only numpy functions. Don't forget regularization!

        w_opt = None
        # ====== YOUR CODE: ======
        # Wopt = (X^t*X + NλI)^-1  * X^t*y
        # I size is DxD
        N = X.shape[0]
        D = X.shape[1]
        regularization = np.eye(D)*self.reg_lambda*N
        regularization[0,0] = 0 #bias
        XtX = np.dot(X.T,X)
        Xty = np.dot(X.T, y)
        w_opt = np.dot(np.linalg.inv(XtX + regularization),(Xty))
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    # TODO: Implement according to the docstring description.
    # ====== YOUR CODE: ======
    y = df.loc[:,target_name].values
#     print(y)
    if feature_names is None:
        X = df.drop(target_name, axis='columns').values
    else:
        X = df.loc[:, feature_names].values
    y_pred = model.fit_predict(X,y)
    # ========================
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ====== 
        xb = np.hstack((np.ones((X.shape[0],1)),X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        # ????????????????????????????????????? ** Do I need something else? ** ??????????????????????????
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        #drop target
#         X_transformed = PolynomialFeatures(self.degree).fit_transform(X)
        poly = sklearn.preprocessing.PolynomialFeatures(degree = self.degree)
        X_transformed = poly.fit_transform(X)
        # use degree = 2 has 91% accuracy
        # use degree = 3 has 99% accuracy?
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    pearson = df.corr(method = 'pearson')[target_feature]
    # find the top n absolute values:
    sort_abs_top_n = abs(pearson).sort_values(ascending = False)[1:n+1] # the first one is MEDV correlation to itself (1.0), remove it
    # print(sort_abs_top_n) # test
    top_n_features = sort_abs_top_n.keys().values
    top_n_corr = (pearson)[top_n_features].values # without abs()
    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    mse = ((y-y_pred)**2).mean()
    # mse = np.power(np.linalg.norm(y - y_pred), 2) / len(y)

    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y - y.mean())**2)
    # ========================
    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    # initial parameters: 'bostonfeaturestransformer__degree': 2, 'linearregressor__reg_lambda': 0.1
    #init:
    min_mse = np.inf
    best_deg = 0
    best_lambda = 0
    # k fold from sklearn
    k_fold = sklearn.model_selection.KFold(n_splits = k_folds)
    for lambda_i in lambda_range:  
        for deg_j in degree_range:
            current_mse = 0 # init
            for train_index, test_index in k_fold.split(X):
                model = model.set_params(**{'bostonfeaturestransformer__degree': deg_j, 'linearregressor__reg_lambda': lambda_i})
                model.fit(X[train_index], y[train_index])
                y_pred = model.predict(X[test_index])
#                 y_pred = model.fit_predict(X[test_index], y[train_index]) # not work ...
                current_mse += mse_score(y[test_index],y_pred)
            if current_mse < min_mse :
                best_deg = deg_j
                best_lambda = lambda_i
                min_mse = current_mse
    best_params = {'bostonfeaturestransformer__degree':best_deg, 'linearregressor__reg_lambda': best_lambda}   
    # ========================

    return best_params
