import numpy as np


class Metrics:

    def mean_squared_error(y, y_hat):
        """
        Computes Mean Squared Error (MSE).

        Parameters:
        - y (np.ndarray): True values.
        - y_hat (np.ndarray): Predicted values.

        Returns:
        - float: Mean Squared Error.
        """
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)
        return np.mean((y - y_hat) ** 2)

    def root_mean_squared_error(y, y_hat):
        """
        Computes Root Mean Squared Error (RMSE) using MSE.

        Parameters:
        - y (np.ndarray): True values.
        - y_hat (np.ndarray): Predicted values.

        Returns:
        - float: Root Mean Squared Error.
        """
        mse = Metrics.mean_squared_error(y, y_hat)
        return np.sqrt(mse)

    def normalized_root_mean_squared_error(y, y_hat):
        """
        Computes Normalized Root Mean Squared Error (NRMSE) using RMSE.

        Parameters:
        - y (np.ndarray): True values.
        - y_hat (np.ndarray): Predicted values.

        Returns:
        - float: Normalized RMSE (RMSE divided by the range of true values).
        """
        rmse = Metrics.root_mean_squared_error(y, y_hat)
        return rmse / (np.max(y) - np.min(y))

    def accuracy(y, y_hat):
        """
        Computes Accuracy.

        Parameters:
        - y (np.ndarray): True values (binary or categorical).
        - y_hat (np.ndarray): Predicted values.

        Returns:
        - float: Accuracy score (fraction of correct predictions).
        """
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)
        return np.mean(y == y_hat)
    
    def absolute_percentage_error(y, y_hat):
        """
        Computes Absolute Percentage Error (APE) for each prediction.

        Returns:
        - np.ndarray: APE values for each prediction.
        """
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)
        return np.abs((y - y_hat) / y) * 100

    def mean_absolute_percentage_error(y, y_hat):
        """
        Computes Mean Absolute Percentage Error (MAPE).

        Returns:
        - float: MAPE value.
        """
        ape = Metrics.absolute_percentage_error(y, y_hat)
        return np.mean(ape)

    def symmetric_mean_absolute_percentage_error(y, y_hat):
        """
        Computes Symmetric Mean Absolute Percentage Error (SMAPE).

        Returns:
        - float: SMAPE value.
        """
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)
        denominator = (np.abs(y) + np.abs(y_hat)) / 2
        smape = np.abs(y - y_hat) / denominator
        return np.mean(smape) * 100
    

