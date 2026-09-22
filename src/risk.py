"""Threshold-based prediction labeling. No Streamlit imports.

The final model's threshold (0.5) is the only classification rule used here.
No LOW/MEDIUM/HIGH business risk tiers are defined -- those would be an
invented operational policy this project has no basis for. If the app
renders a probability bar with color bands, that rendering is a
visualization choice made in app.py, and must be labeled there as
illustrative only, not as this module's output.
"""

DEFAULT_THRESHOLD = 0.5

FRAUD_LABEL = "FRAUD"
LEGITIMATE_LABEL = "LEGITIMATE"


def classify(probability: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """probability >= threshold -> FRAUD, else LEGITIMATE. This is the same
    fixed rule used for every reported metric in Steps 5-10 -- not tuned or
    reinterpreted here."""
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    return FRAUD_LABEL if probability >= threshold else LEGITIMATE_LABEL
