import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error,root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

class Get_Regression_Metrics:
    """
    A class to compute and display regression evaluation metrics for 
    both training and testing datasets.

    Metrics included:
    - Mean Absolute Error (MAE)
    - Root Mean Square Error (RMSE)
    - R² (Coefficient of Determination)
    - Mean Absolute Percentage Error (MAPE)
    - Symmetrical Mean Absolute Percentage Error (sMAPE)
    """
    
    def __init__(self, y_train, y_train_pred, y_test, y_test_pred):
        """
        Initializes Get_Regression_Metrics class with a actual and predicted values for test and train.

        :param y_train: Actual target values for the training set.
        :param y_train_pred: Predicted target values for the training set.
        :param y_test: Actual target values for the testing set.
        :param y_test_pred: Predicted target values for the testing set.
        """
        self.y_train = y_train
        self.y_train_pred = y_train_pred
        self.y_test = y_test
        self.y_test_pred = y_test_pred

    def get_mae(self):
        """
        Mean absolute error (MAE) is an average of the absolute errors.
        MAE is always greater than or equal to 0 and MAE value 0 represents perfect fit.
        The MAE units are the same as the predicted target.  
        """
        self.mae_train = mean_absolute_error(self.y_train, self.y_train_pred)
        self.mae_test = mean_absolute_error(self.y_test, self.y_test_pred)
        print(f"MAE(train): {self.mae_train:.3f}")
        print(f"MAE(test): {self.mae_test:.3f}")

    def get_rmse(self):
        """
        Root Mean Square Error (RMSE) represents the square root of the variance of the residuals.
        The smaller the RMSE, the closer your model's predictions are to reality.
        RMSE is expressed in the same unit as the predicted values 
        """ 
        self.rmse_train = root_mean_squared_error(self.y_train, self.y_train_pred)
        self.rmse_test = root_mean_squared_error(self.y_test, self.y_test_pred)
        print(f"RMSE(train): {self.rmse_train:.3f}")
        print(f"RMSE(test): {self.rmse_test:.3f}")

    def get_r2(self):
        """
        R² (coefficient of determination) regression score function.
        R² is an element of [0, 1].
        Best possible score is 1.0 and it can be negative (the model can be arbitrarily worse).
        """
        self.r2_train = r2_score(self.y_train, self.y_train_pred)
        self.r2_test = r2_score(self.y_test, self.y_test_pred)
        print(f"R²(train): {self.r2_train:.3f}")
        print(f"R²(test): {self.r2_test:.3f}")

    def get_mape(self):
        """
        Mean absolute percentage error (MAPE) regression loss. 
        It measures accuracy as a percentage.
        Lower values of MAPE indicate higher accuracy.
        """
        self.mape_train = mean_absolute_percentage_error(self.y_train, self.y_train_pred)
        self.mape_test = mean_absolute_percentage_error(self.y_test, self.y_test_pred)
        print(f"MAPE (train): {self.mape_train:.3f}")
        print(f"MAPE (test): {self.mape_test:.3f}")

    def get_smape(self):
        """
        Symmetrical mean absolute percentage error (sMAPE) 
        is an accuracy measure based on percentage (or relative) errors.
        The resulting score ranges between 0 and 1, where a score of 0 indicates a perfect match. 
        """
        def smape(actual, forecast):
            actual = np.array(actual)
            forecast = np.array(forecast)
            smape = np.mean(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))) * 100
            return smape
        self.smape_train = smape(self.y_train, self.y_train_pred)
        self.smape_test = smape(self.y_test, self.y_test_pred)
        print(f"sMAPE (train): {self.smape_train:.3f}")
        print(f"sMAPE (test): {self.smape_test:.3f}")

    def get_all_metrics(self):
        """
        Get all the metrics.
        """
        self.get_mae()
        self.get_rmse()
        self.get_r2()
        self.get_mape()
        self.get_smape()

class SplitData:
    """
    The SplitData class splits datasets into training and testing sets. 
    """
    def __init__(self, dataframe, target_column, drop_columns=None, test_size=0.2, random_state=42):
        """
        Initialize the SplitData class.

        :param dataframe: The pandas DataFrame containing the dataset.
        :param target_column: The name of the target column to predict.
        :param drop_columns: List of columns to drop from features. Default is None.
        :param test_size: Proportion of the dataset to include in the test split. Default is 0.2 (20%)
        :param random_state: Random state for reproducibility. Default is 42.
        """
        self.dataframe = dataframe
        self.target_column = target_column
        self.drop_columns = drop_columns if drop_columns else []
        self.test_size = test_size
        self.random_state = random_state

    def split(self):
        """
        Splits the data into training and testing sets.

        :return: X_train: Training features.
        :return: X_test: Testing features.
        :return: y_train: Training target values.
        :return: y_test: Testing target values.
        """
        # Drop specified columns and separate target
        X = self.dataframe.drop(columns=[self.target_column] + self.drop_columns, axis=1)
        y = self.dataframe[self.target_column].values.reshape(-1, 1)

        # Apply square root transformation to the target
        y_transformed = np.sqrt(y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, test_size=self.test_size, random_state=self.random_state
        )
        print("Database split into train and test sets with an 80/20 ratio")
        return X_train, X_test, y_train, y_test

class ModelDecisionTreeRegressor:
    def __init__(self, random_state=100, min_samples_split=15, min_samples_leaf=10, max_leaf_nodes=150, max_depth=20):
        """
        Initialize the ModelDecisionTreeRegressor class.

        :param random_state: Random state for reproducibility.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
        :param max_leaf_nodes: Maximum number of leaf nodes.
        :param max_depth: Maximum depth of the tree.
        """
        self.model = DecisionTreeRegressor(
            random_state=random_state,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth
        )

    def train(self, X_train, y_train):
        """
        Train the Decision Tree Regressor model.

        :param X_train: Training features.
        :param y_train: Training target values.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions using the trained model.

        :param X: Features to predict on.
        :return Predictions from the model.
        """
        return self.model.predict(X)

    def evaluate(self, X_train, X_test, y_train, y_test):
        """
        Evaluate the models performance and calculate metrics.

        :param X_train: Training features.
        :param X_test: Testing features.
        :param y_train: Training target values.
        :param y_test: Testing target values.
        :return A dictionary of metrics for both training and testing sets.
        """
        # Predictions
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)

        # Transform predictions and actual values back to the original scale (squared)
        y_train_pred_original = y_train_pred ** 2
        y_test_pred_original = y_test_pred ** 2
        y_train_original = y_train ** 2
        y_test_original = y_test ** 2

        # Get metrics
        metrics = Get_Regression_Metrics(
            y_train_original, y_train_pred_original,
            y_test_original, y_test_pred_original
        )
        print(f'Decision Tree Regressor metrics:{metrics}')
        return metrics.get_all_metrics()

    def plot_tree(self,X_train):
        """
        Get the visualization for the decision tree regressor model. 
        """
        visualization_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")
        plt.figure(figsize=(100, 30))
        plot_tree(self.model, filled=True, feature_names=X_train.columns, rounded=True, fontsize=12)
        plt.title('Decision Tree Regressor Visualization')
        plt.savefig(os.path.join(visualization_dir, "decision_tree.png"))
        plt.close()
        print("Decision tree plot saved to visualization folder.")

class ModelRandomForestRegressor:
    def __init__(self, random_state=100, n_estimators=150, min_samples_split=100, min_samples_leaf=17, max_leaf_nodes=100, max_depth=100):
        """
        Initialize the ModelRandomForestRegressor class.

        :param random_state: Random state for reproducibility.
        :param n_estimators: The number of trees in the forest.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
        :param max_leaf_nodes: Maximum number of leaf nodes.
        :param max_depth: Maximum depth of the trees.
        """
        self.model = RandomForestRegressor(
            random_state=random_state,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth
        )

    def train(self, X_train, y_train):
        """
        Train the Random Forest Regressor model.

        :param X_train: Training features.
        :param y_train: Training target values.
        """
        self.model.fit(X_train, y_train.ravel()) 

    def predict(self, X):
        """
        Make predictions using the trained model.

        :param X: Features to predict on.
        :return Predictions from the model.
        """
        return self.model.predict(X)

    def evaluate(self, X_train, X_test, y_train, y_test):
        """
        Evaluate the models performance and calculate metrics.

        :param X_train: Training features.
        :param X_test: Testing features.
        :param y_train: Training target values.
        :param y_test: Testing target values.
        :return A dictionary of metrics for both training and testing sets.
        """
        # Flatten y_train and y_test to 1D arrays
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        # Predictions
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)

        # Transform predictions and actual values back to the original scale (squared)
        y_train_pred_original = y_train_pred ** 2
        y_test_pred_original = y_test_pred ** 2
        y_train_original = y_train ** 2
        y_test_original = y_test ** 2

        # Get metrics
        metrics = Get_Regression_Metrics(
            y_train_original, y_train_pred_original,
            y_test_original, y_test_pred_original
        )
        print(f'Random Forest Regressor metrics: {metrics}')
        return metrics.get_all_metrics()
    
    def plot_trees(self, X_train):
        """
        Get the visualization for the random forest regressor model. 
        Only the first 3 trees will be saved.
        """
        visualization_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")
        if not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)

        num_trees_to_plot = min(3, len(self.model.estimators_))

        for idx, tree in enumerate(self.model.estimators_[:num_trees_to_plot]):
            plt.figure(figsize=(100, 20))
            plot_tree(tree, filled=True, feature_names=X_train.columns, rounded=True, fontsize=12)
            plt.title(f'Random Forest Regressor Tree {idx + 1}')
            plt.savefig(os.path.join(visualization_dir, f"random_forest_tree_{idx + 1}.png"))
            plt.close()

        print(f"First {num_trees_to_plot} random forest trees saved to visualization folder.")