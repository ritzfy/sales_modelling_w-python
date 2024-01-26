# --- 1) IMPORTING PACKAGES
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load data
def load_data(path: str = "data.csv") -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame.

    :param path: Path to the CSV file (optional, defaults to current working directory).
    :return: DataFrame containing the loaded data.
    """

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise ValueError(f"Data file not found at path: {path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {path}")
    except pd.errors.ParserError:
        raise ValueError(f"Failed to parse CSV file: {path}")

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    else:
        warnings.warn("Column 'Unnamed: 0' not found in the DataFrame.")

    return df

# --- 2) DEFINE GLOBAL CONSTANTS
K = 10
SPLIT = 0.75

# --- 3) ALGORITHM CODE

# Create target variable and predictor variables
def create_target_and_predictors(
    data: pd.DataFrame = None, 
    target: str = "estimated_stock_pct"
):
    """
    This function takes in a Pandas DataFrame and splits the columns
    into a target column and a set of predictor variables, i.e. X & y.
    These two splits of the data will be used to train a supervised 
    machine learning model.

    :param      data: pd.DataFrame, dataframe containing data for the 
                      model
    :param      target: str (optional), target variable that you want to predict

    :return     X: pd.DataFrame
                y: pd.Series
    """

    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# Train algorithm
def train_algorithm_with_cross_validation(
    X: pd.DataFrame = None, 
    y: pd.Series = None
):
    """
    This function takes the predictor and target variables and
    trains a Random Forest Regressor model across K folds. Using
    cross-validation, performance metrics will be output for each
    fold during training.

    :param      X: pd.DataFrame, predictor variables
    :param      y: pd.Series, target variable

    :return
    """

    # Create a list that will store the accuracies of each fold
    accuracy = []

    # Enter a loop to run K folds of cross-validation
    for fold in range(0, K):

        # Instantiate algorithm and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        # Scale X data, we scale the data because it helps the algorithm to converge
        # and helps the algorithm to not be greedy with large values
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Finish by computing the average MAE across all folds
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")
    return model.feature_importances_

 
# --- 4) MAIN FUNCTION
def main():
    """
    This main function loads the data, prepares it for modeling,
    trains the algorithm, and visualizes the results.
    """

    # Load the data
    try:
        data = load_data()
    except FileNotFoundError:
        print("Error: Data file not found. Please check the path and try again.")
        return

    # Creating target and predictor variables
    X, y = create_target_and_predictors(data)

    features = [i.split("__")[0] for i in X.columns]  # Extract feature names
    # Training the algorithm with cross-validation and getting important scores from the trained model
    importances = train_algorithm_with_cross_validation(X, y)
    indices = np.argsort(importances)  # Sort indices based on importance

    # Visualizing the results
    plt.figure(figsize=(10, 20))  # Create the plot
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


if __name__ == "__main__":
   main()