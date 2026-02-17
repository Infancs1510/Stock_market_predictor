import sys
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as metrics

def run_baseline_regression(df: pd.DataFrame) -> None:
    X = df[["Open", "High", "Low", "Volume"]]
    y = df["Close"]
    # Time-based split (no leakage)
    df = df.sort_values("Date").reset_index(drop=True)

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[["Open", "High", "Low", "Volume"]]
    y_train = train_df["Close"]

    X_test = test_df[["Open", "High", "Low", "Volume"]]
    y_test = test_df["Close"]


    model = lm.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = metrics.r2_score(y_test, y_pred)
    print(f"Baseline R² Score: {r2}")
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    rmse = (metrics.mean_squared_error(y_test, y_pred)) ** 0.5

    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")

    print("Linear Regression Model Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    coef = pd.Series(model.coef_, index=X.columns)
    print("\nFeature Coefficients:")
    print(coef)








def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans common issues:
    - Removes header-like junk rows (e.g., rows where numeric columns contain 'AAPL')
    - Converts Date to datetime
    - Converts price/volume columns to numeric
    - Drops rows that become invalid after conversion
    """

    # Standardize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # If expected columns exist, clean them
    expected_cols = {"Date", "Close", "High", "Low", "Open", "Volume"}
    if expected_cols.issubset(set(df.columns)):
        # Remove obvious junk rows where "Date" looks like ticker text (e.g., "AAPL")
        df = df[df["Date"].astype(str).str.upper() != "AAPL"]

        # Convert Date to datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Convert numeric columns
        numeric_columns = ["Close", "High", "Low", "Open", "Volume"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows that became invalid
        df = df.dropna(subset=["Date"] + numeric_columns)

    return df


def top_correlations(df: pd.DataFrame, top_n: int = 5):
    """
    Returns top absolute correlations among numeric columns (excluding self-correlation).
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return None

    corr_matrix = numeric_df.corr(numeric_only=True)

    corr_pairs = (
        corr_matrix.abs()
        .unstack()
        .sort_values(ascending=False)
    )

    # Remove self-correlation and duplicates (A-B and B-A)
    corr_pairs = corr_pairs[corr_pairs < 1]
    corr_pairs = corr_pairs[~corr_pairs.index.duplicated(keep="first")]

    return corr_pairs.head(top_n)


def analyze_data(file_path: str):
    try:
        df = pd.read_csv(file_path)

        print("\nDataset Loaded Successfully!")
        print(f"Raw Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

        # Clean data
        df = clean_dataframe(df)

        print("After Cleaning:")
        print(f"Clean Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

        print("First 5 Rows (Cleaned):")
        print(df.head(), "\n")

        print("Column Data Types (Cleaned):")
        print(df.dtypes, "\n")

        print("Missing Values Per Column (Cleaned):")
        print(df.isnull().sum(), "\n")

        print("Duplicate Rows (Cleaned):", df.duplicated().sum(), "\n")

        # Numeric summary
        numeric_df = df.select_dtypes(include=["number"])
        if not numeric_df.empty:
            print("Summary Statistics (Numeric Columns):")
            print(numeric_df.describe(), "\n")
        else:
            print("Summary Statistics (Numeric Columns): None found.\n")

        # Categorical summary
        categorical_df = df.select_dtypes(exclude=["number", "datetime"])
        print("Categorical Columns Summary (Top 5 values):")
        if not categorical_df.empty:
            for col in categorical_df.columns:
                print(f"\nColumn: {col}")
                print(categorical_df[col].astype(str).value_counts().head(5))
            print()
        else:
            print("No categorical columns found.\n")

        # Correlations
        tc = top_correlations(df, top_n=5)
        if tc is not None:
            print("Top 5 Strongest Correlations (absolute value):")
            print(tc, "\n")
        else:
            print("Correlations: Not enough numeric columns to compute.\n")

        run_baseline_regression(df)
        print("Analysis complete.\n")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        print("The file is empty or not a valid CSV.")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <csv_file>")
        print("Example: python analyze.py data.csv")
        

    else:
        analyze_data(sys.argv[1])
        #run_baseline_regression(df)

