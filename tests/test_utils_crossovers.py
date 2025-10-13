import pandas as pd
import numpy as np

from quantjourney_ti._utils import detect_crossovers


def test_detect_crossovers_no_off_by_one():
    # Construct two series where s1 crosses above s2 at index 3 exactly
    idx = pd.RangeIndex(0, 7)
    s2 = pd.Series([0, 0, 0, 0, 0, 0, 0], index=idx, dtype=float)
    s1 = pd.Series([-2, -1, 0, 1, 2, 1, 0], index=idx, dtype=float)

    crosses = detect_crossovers(s1, s2)
    # Expect bullish at index 3 only
    assert crosses.loc[3, "bullish"] == 1
    assert crosses["bullish"].sum() == 1
    assert crosses["bearish"].sum() == 0

    # Now invert to force a bearish cross at index 3
    crosses2 = detect_crossovers(s2, s1)
    assert crosses2.loc[3, "bearish"] == 1
    assert crosses2["bearish"].sum() == 1
    assert crosses2["bullish"].sum() == 0

