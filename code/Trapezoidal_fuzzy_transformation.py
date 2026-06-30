import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# 1. Trapezoidal membership function


def trapezoidal_membership(x, a, b, c, d):
    """
    Trapezoidal fuzzy membership function.

    a = left lower boundary
    b = left upper boundary
    c = right upper boundary
    d = right lower boundary

    The value is:
    0 before a and after d
    1 between b and c
    gradually increasing between a and b
    gradually decreasing between c and d
    """

    if pd.isna(x):
        return 0.0

    if x <= a or x >= d:
        return 0.0

    if a < x < b:
        return (x - a) / (b - a) if b != a else 1.0

    if b <= x <= c:
        return 1.0

    if c < x < d:
        return (d - x) / (d - c) if d != c else 1.0

    return 0.0



# 2. Create trapezoidal parameters


def create_trapezoidal_parameters(df, feature_name, mode="balanced"):
    """
    Create trapezoidal fuzzy parameters for one feature.

    mode = "balanced"
        Creates Low, Medium, and High categories using quantiles.

    mode = "age"
        Creates Young, Middle, and Old categories.
    """

    values = df[feature_name].dropna()

    vmin = float(values.min())
    vmax = float(values.max())

    q20 = float(values.quantile(0.20))
    q25 = float(values.quantile(0.25))
    q40 = float(values.quantile(0.40))
    q50 = float(values.quantile(0.50))
    q60 = float(values.quantile(0.60))
    q75 = float(values.quantile(0.75))
    q80 = float(values.quantile(0.80))

    if mode == "age":
        params = {
            "Young":  (vmin, vmin, q25, q50),
            "Middle": (q25, q40, q60, q75),
            "Old":    (q50, q75, vmax, vmax)
        }

    else:
        params = {
            "Low":    (vmin, vmin, q25, q50),
            "Medium": (q25, q40, q60, q75),
            "High":   (q50, q75, vmax, vmax)
        }

    return params



# 3. Assign maximum membership label


def get_max_trapezoidal_label(x, params):
    """
    Calculate membership for all fuzzy classes
    and return the class with the highest membership.
    """

    memberships = {}

    for label, points in params.items():
        a, b, c, d = points
        memberships[label] = trapezoidal_membership(x, a, b, c, d)

    best_label = max(memberships, key=memberships.get)

    if memberships[best_label] == 0:
        return "None"

    return best_label



# 4. Transform one feature


def transform_trapezoidal_feature(df, feature_name, output_column=None, mode="balanced"):
    """
    Transform one numerical feature into one fuzzy categorical feature.

    Example:
    age      -> trap_age
    chol     -> trap_chol
    trestbps -> trap_trestbps
    thalach  -> trap_thalach
    """

    df = df.copy()

    if output_column is None:
        output_column = "trap_" + feature_name

    params = create_trapezoidal_parameters(
        df=df,
        feature_name=feature_name,
        mode=mode
    )

    df[output_column] = df[feature_name].apply(
        lambda x: get_max_trapezoidal_label(x, params)
    )

    print("\nFeature transformed:", feature_name)
    print("New column created :", output_column)

    print("\nParameters used:")
    for label, values in params.items():
        print(label, ":", values)

    print("\nTransformed value distribution:")
    print(df[output_column].value_counts(dropna=False))

    return df, params



# 5. Transform multiple features


def transform_multiple_trapezoidal_features(df, feature_settings):
    """
    Apply trapezoidal fuzzy transformation to multiple features.

    Example feature_settings:

    {
        "age": {"output": "trap_age", "mode": "age"},
        "chol": {"output": "trap_chol", "mode": "balanced"},
        "trestbps": {"output": "trap_trestbps", "mode": "balanced"},
        "thalach": {"output": "trap_thalach", "mode": "balanced"}
    }
    """

    df = df.copy()
    all_params = {}

    for feature_name, settings in feature_settings.items():

        output_column = settings.get("output", "trap_" + feature_name)
        mode = settings.get("mode", "balanced")

        df, params = transform_trapezoidal_feature(
            df=df,
            feature_name=feature_name,
            output_column=output_column,
            mode=mode
        )

        all_params[feature_name] = params

    return df, all_params



# 6. Plot trapezoidal membership functions


def plot_trapezoidal_membership(feature_name, params, x_min=None, x_max=None):
    """
    Plot trapezoidal membership curves for one feature.
    """

    all_points = []

    for points in params.values():
        all_points.extend(points)

    if x_min is None:
        x_min = min(all_points)

    if x_max is None:
        x_max = max(all_points)

    x_values = np.linspace(x_min, x_max, 1000)

    plt.figure(figsize=(10, 5))

    for label, points in params.items():
        a, b, c, d = points

        y_values = [
            trapezoidal_membership(x, a, b, c, d)
            for x in x_values
        ]

        plt.plot(x_values, y_values, label=label)

    plt.xlabel(feature_name)
    plt.ylabel("Membership value")
    plt.title("Trapezoidal Fuzzy Membership Function for " + feature_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# 7. Example use


if __name__ == "__main__":

    # Load your original dataset
    df = pd.read_csv("heart.csv")

    # Select the numerical features to transform
    feature_settings = {
        "age": {
            "output": "trap_age",
            "mode": "age"
        },
        "chol": {
            "output": "trap_chol",
            "mode": "balanced"
        },
        "trestbps": {
            "output": "trap_trestbps",
            "mode": "balanced"
        },
        "thalach": {
            "output": "trap_thalach",
            "mode": "balanced"
        }
    }

    # Apply trapezoidal fuzzy transformation
    transformed_df, all_params = transform_multiple_trapezoidal_features(
        df=df,
        feature_settings=feature_settings
    )

    # Save transformed dataset
    transformed_df.to_csv("heart_trapezoidal_fuzzy.csv", index=False)

    print("\nTrapezoidal fuzzy transformation completed.")
    print("Saved file: heart_trapezoidal_fuzzy.csv")

    # Visualize one feature
    plot_trapezoidal_membership(
        feature_name="thalach",
        params=all_params["thalach"]
    )
