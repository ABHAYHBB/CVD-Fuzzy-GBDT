import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Basic membership functions

def gaussian_membership(x, center, sigma):
    """
    Gaussian membership function.
    It gives the highest membership near the clinical center value.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def sigmoid_membership(x, center, slope):
    """
    Sigmoid membership function.
    It is mainly used to create a smooth transition near class boundaries.
    """
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-(x - center) / slope))


def lower_hybrid_membership(x, c_lower, sigma_lower, c_middle, transition_slope):
    """
    Hybrid lower-category membership.

    The lower region is kept strong before c_lower.
    After that, Gaussian decay is used, and the sigmoid term helps the curve
    reduce smoothly as the feature moves toward the middle region.
    """
    x = np.asarray(x, dtype=float)

    gaussian_part = np.where(
        x <= c_lower,
        1.0,
        gaussian_membership(x, c_lower, sigma_lower)
    )

    sigmoid_gate = 1 - sigmoid_membership(x, c_middle, transition_slope)

    return gaussian_part * sigmoid_gate


def middle_hybrid_membership(x, c_middle, sigma_middle, c_lower, c_upper, transition_slope):
    """
    Hybrid middle-category membership.

    Gaussian membership gives the central peak.
    Two sigmoid gates keep the middle class active mainly between the lower
    and upper clinical boundaries.
    """
    x = np.asarray(x, dtype=float)

    gaussian_part = gaussian_membership(x, c_middle, sigma_middle)

    left_gate = sigmoid_membership(x, c_lower, transition_slope)
    right_gate = 1 - sigmoid_membership(x, c_upper, transition_slope)

    return gaussian_part * left_gate * right_gate


def upper_hybrid_membership(x, c_upper, sigma_upper, c_middle, transition_slope):
    """
    Hybrid upper-category membership.

    The upper region is kept strong after c_upper.
    Before that, Gaussian increase is used, and the sigmoid term helps the
    curve rise smoothly from the middle region.
    """
    x = np.asarray(x, dtype=float)

    gaussian_part = np.where(
        x >= c_upper,
        1.0,
        gaussian_membership(x, c_upper, sigma_upper)
    )

    sigmoid_gate = sigmoid_membership(x, c_middle, transition_slope)

    return gaussian_part * sigmoid_gate



# Combined fuzzy transformation

def calculate_fuzzy_memberships(x, params, normalize=False):
    """
    Calculate Low, Medium, and High membership values for one feature.

    Parameters
    ----------
    x : array-like
        Feature values.
    params : dict
        Dictionary containing clinical/fuzzy parameters.
    normalize : bool
        If True, Low + Medium + High is normalized to 1 for each sample.

    Returns
    -------
    low, medium, high : arrays
        Membership values for the three fuzzy categories.
    """

    low = lower_hybrid_membership(
        x,
        c_lower=params["c_lower"],
        sigma_lower=params["sigma_lower"],
        c_middle=params["c_middle"],
        transition_slope=params["transition_slope"]
    )

    medium = middle_hybrid_membership(
        x,
        c_middle=params["c_middle"],
        sigma_middle=params["sigma_middle"],
        c_lower=params["c_lower"],
        c_upper=params["c_upper"],
        transition_slope=params["transition_slope"]
    )

    high = upper_hybrid_membership(
        x,
        c_upper=params["c_upper"],
        sigma_upper=params["sigma_upper"],
        c_middle=params["c_middle"],
        transition_slope=params["transition_slope"]
    )

    if normalize:
        total = low + medium + high
        total = np.where(total == 0, 1, total)

        low = low / total
        medium = medium / total
        high = high / total

    return low, medium, high


def transform_single_feature(df, feature_name, params, normalize=False):
    """
    Transform one numerical feature into three fuzzy membership columns.

    Example:
    thalach becomes:
    thalach_low
    thalach_medium
    thalach_high
    thalach_fuzzy_label
    """
    values = df[feature_name].astype(float).values

    low, medium, high = calculate_fuzzy_memberships(
        values,
        params=params,
        normalize=normalize
    )

    transformed = pd.DataFrame(index=df.index)

    transformed[f"{feature_name}_low"] = low
    transformed[f"{feature_name}_medium"] = medium
    transformed[f"{feature_name}_high"] = high

    membership_matrix = np.vstack([low, medium, high]).T
    labels = np.array(["Low", "Medium", "High"])

    transformed[f"{feature_name}_fuzzy_label"] = labels[np.argmax(membership_matrix, axis=1)]

    return transformed


def transform_dataset(df, fuzzy_params, normalize=False, keep_original=True):
    """
    Apply fuzzy transformation to all selected numerical features.

    fuzzy_params should be a dictionary where each key is a feature name.
    """
    output_df = df.copy() if keep_original else pd.DataFrame(index=df.index)

    for feature_name, params in fuzzy_params.items():
        if feature_name not in df.columns:
            raise ValueError(f"Feature '{feature_name}' is not present in the dataframe.")

        feature_fuzzy_df = transform_single_feature(
            df=df,
            feature_name=feature_name,
            params=params,
            normalize=normalize
        )

        output_df = pd.concat([output_df, feature_fuzzy_df], axis=1)

    return output_df


# Visualization


def plot_fuzzy_membership(feature_name, params, x_min=None, x_max=None, normalize=False):
    """
    Plot Low, Medium, High, and Sum membership curves for one feature.
    """

    if x_min is None:
        x_min = params["c_lower"] - 3 * params["sigma_lower"]

    if x_max is None:
        x_max = params["c_upper"] + 3 * params["sigma_upper"]

    x = np.linspace(x_min, x_max, 1000)

    low, medium, high = calculate_fuzzy_memberships(
        x,
        params=params,
        normalize=normalize
    )

    total = low + medium + high

    plt.figure(figsize=(12, 6))

    plt.plot(x, low, label="Low")
    plt.plot(x, medium, label="Medium")
    plt.plot(x, high, label="High")
    plt.plot(x, total, label="Sum", linestyle="--")

    plt.axvline(params["c_lower"], linestyle="--", linewidth=1)
    plt.axvline(params["c_middle"], linestyle="--", linewidth=1)
    plt.axvline(params["c_upper"], linestyle="--", linewidth=1)

    plt.axhline(y=1, linestyle="--", linewidth=1)

    plt.xlabel(feature_name)
    plt.ylabel("Membership value")
    plt.title(f"Hybrid Gaussian-Sigmoid Fuzzy Membership Functions for {feature_name}")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Example use

if __name__ == "__main__":

    # Example feature: maximum heart rate achieved.
    # Replace these values with the exact parameters reported in your manuscript table.
    fuzzy_params = {
        "thalach": {
            "c_lower": 100,
            "sigma_lower": 10,
            "c_middle": 130,
            "sigma_middle": 10,
            "c_upper": 160,
            "sigma_upper": 10,
            "transition_slope": 6
        }
    }

    # Visualize the membership curves first.
    plot_fuzzy_membership(
        feature_name="Maximum Heart Rate Achieved",
        params=fuzzy_params["thalach"],
        x_min=80,
        x_max=180,
        normalize=False
    )

    # Small example data to check whether transformation is working correctly.
    sample_df = pd.DataFrame({
        "thalach": [88, 100, 115, 130, 145, 160, 178]
    })

    transformed_df = transform_dataset(
        df=sample_df,
        fuzzy_params=fuzzy_params,
        normalize=False,
        keep_original=True
    )

    print("\nFuzzy-transformed sample data:")
    print(transformed_df)
