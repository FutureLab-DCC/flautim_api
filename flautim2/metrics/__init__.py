import numpy as np

def mean_squared_error(y, y_hat):
    """
    Computes Mean Squared Error (MSE).

    Parameters:
    - y (np.ndarray): True values.
    - y_hat (np.ndarray): Predicted values.

    Returns:
    - float: Mean Squared Error.
    """
    y = np.asarray(y).reshape(-1, 1)
    y_hat = np.asarray(y_hat).reshape(-1, 1)
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
    mse = mean_squared_error(y, y_hat)
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
    rmse = root_mean_squared_error(y, y_hat)
    return rmse / (np.max(y) - np.min(y))
