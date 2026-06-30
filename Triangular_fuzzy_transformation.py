import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# 1. Triangular membership function


def triangular_membership(x, a, b, c):
    """
    Basic triangular fuzzy membership function.

    a = left boundary
    b = peak point
    c = right boundary

    Membership is:
    0 outside the triangle
    1 at the peak
    between 0 and 1 inside the triangle
    """

    if pd.isna(x):
        return 0.0

    if x <= a or x >= c:
        return 0.0

    if a < x <= b:
        return (x - a) / (b - a) if b != a else 0.0

    if b < x < c:
        return (c - x) / (c - b) if c != b else 0.0

    return 0.0



# 2. Create triangular parameters automatically


def create_triangular_parameters(df, feature_name, mode="balanced"):
    """
    Create triangular fuzzy parameters from the data.

    mode = "balanced"
        Creates Low, Medium, High categories using q33, q50, q66.

    mode = "age"
        Creates Young, Middle, Old categories with a narrower middle region.
        This follows the logic used in your reference notebook for age.
    """

    clean_values = df[feature_name].dropna()

    vmin = float(clean_values.min())
    vmax = float(clean_values.max())

    if mode == "age":
        q25 = float(clean_values.quantile(0.25))
        q40 = float(clean_values.quantile(0.40))
        q50 = float(clean_values.quantile(0.50))
        q60 = float(clean_values.quantile(0.60))
        q75 = float(clean_values.quantile(0.75))

        params = {
            "Young": (vmin, q25, q50),
            "Middle": (q40, q50, q60),
            "Old": (q50, q75, vmax)
        }

    else:
        q33 = float(clean_values.quantile(0.33))
        q50 = float(clean_values.quantile(0.50))
        q66 = float(clean_values.quantile(0.66))

        params = {
            "Low": (vmin, q33, q50),
            "Medium": (q33, q50, q66),
            "High": (q50, q66, vmax)
        }

    return params


# 3. Assign one fuzzy category to each value


def get_max_membership_label(x, params):
    """
    For one value, calculate all triangular memberships
    and return the category with the highest membership.
    """

    memberships = {}

    for label, points in params.items():
        a, b, c = points
        memberships[label] = triangular_membership(x, a, b, c)

    best_label = max(memberships, key=memberships.get)

    if memberships[best_label] == 0:
        return "None"

    return best_label



# 4. Transform one feature


def transform_triangular_feature(df, feature_name, output_column=None, mode="balanced"):
    """
    Transform one numerical feature into one fuzzy categorical feature.

    Example:
    age      -> max_age
    chol     -> max_chol
    trestbps -> max_trestbps
    thalach  -> max_thalach
    """

    df = df.copy()

    if output_column is None:
        output_column = "max_" + feature_name

    params = create_triangular_parameters(
        df=df,
        feature_name=feature_name,
        mode=mode
    )

    df[output_column] = df[feature_name].apply(
        lambda x: get_max_membership_label(x, params)
    )

    print("\nFeature transformed:", feature_name)
    print("New column created :", output_column)

    print("\nParameters used:")
    for label, values in params.items():
        print(label, ":", values)

    print("\nDistribution:")
    print(df[output_column].value_counts(dropna=False))

    return df, params


# 5. Transform multiple features together


def transform_multiple_triangular_features(df, feature_settings):
    """
    Apply triangular fuzzy transformation to multiple features.

    feature_settings example:

    {
        "age": {"output": "max_age", "mode": "age"},
        "chol": {"output": "max_chol", "mode": "balanced"},
        "trestbps": {"output": "max_trestbps", "mode": "balanced"},
        "thalach": {"output": "max_thalach", "mode": "balanced"}
    }
    """

    df = df.copy()
    all_params = {}

    for feature_name, settings in feature_settings.items():

        output_column = settings.get("output", "max_" + feature_name)
        mode = settings.get("mode", "balanced")

        df, params = transform_triangular_feature(
            df=df,
            feature_name=feature_name,
            output_column=output_column,
            mode=mode
        )

        all_params[feature_name] = params

    return df, all_params


# 6. Plot triangular membership functions


def plot_triangular_membership(feature_name, params, x_min=None, x_max=None):
    """
    Plot triangular fuzzy membership curves for one feature.
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
        a, b, c = points

        y_values = [
            triangular_membership(x, a, b, c)
            for x in x_values
        ]

        plt.plot(x_values, y_values, label=label)

    plt.xlabel(feature_name)
    plt.ylabel("Membership value")
    plt.title("Triangular Fuzzy Membership Function for " + feature_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 7. Example use


if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("heart.csv")

    # Features to transform
    feature_settings = {
        "age": {
            "output": "max_age",
            "mode": "age"
        },
        "chol": {
            "output": "max_chol",
            "mode": "balanced"
        },
        "trestbps": {
            "output": "max_trestbps",
            "mode": "balanced"
        },
        "thalach": {
            "output": "max_thalach",
            "mode": "balanced"
        }
    }

    # Apply triangular fuzzy transformation
    transformed_df, all_params = transform_multiple_triangular_features(
        df=df,
        feature_settings=feature_settings
    )

    # Save transformed dataset
    transformed_df.to_csv("heart_triangular_fuzzy.csv", index=False)

    print("\nTriangular fuzzy transformation completed.")
    print("Saved file: heart_triangular_fuzzy.csv")

    # Visualize one transformed feature
    plot_triangular_membership(
        feature_name="thalach",
        params=all_params["thalach"]
    )
