#test_funding.py

import math
import pytest
from execution_policy import calculate_continuous_funding

def test_zero_elapsed_is_zero():
    assert calculate_continuous_funding("BUY", 10.0, 100.0, 0.001, 0) == 0.0

def test_zero_funding_is_zero():
    assert calculate_continuous_funding("BUY", 10.0, 100.0, 0.0, 60_000) == 0.0

def test_positive_funding_long_pays():
    result = calculate_continuous_funding("BUY", 1.0, 100.0, 0.001, 8 * 3600 * 1000)
    assert result > 0

def test_positive_funding_short_receives():
    result = calculate_continuous_funding("SELL", 1.0, 100.0, 0.001, 8 * 3600 * 1000)
    assert result < 0

def test_negative_funding_long_receives():
    result = calculate_continuous_funding("BUY", 1.0, 100.0, -0.001, 8 * 3600 * 1000)
    assert result < 0

def test_negative_funding_short_pays():
    result = calculate_continuous_funding("SELL", 1.0, 100.0, -0.001, 8 * 3600 * 1000)
    assert result > 0

def test_exact_eight_hour_amount():
    result = calculate_continuous_funding("BUY", 1.0, 100.0, 0.001, 8 * 3600 * 1000)
    assert math.isclose(result, 0.10, rel_tol=0.0, abs_tol=1e-12)

def test_funding_scales_linearly_with_elapsed_time():
    full_8h = calculate_continuous_funding("BUY", 1.0, 100.0, 0.001, 8 * 3600 * 1000)
    half_8h = calculate_continuous_funding("BUY", 1.0, 100.0, 0.001, 4 * 3600 * 1000)
    one_hour = calculate_continuous_funding("BUY", 1.0, 100.0, 0.001, 3600 * 1000)
    assert math.isclose(half_8h, full_8h * 0.5)
    assert math.isclose(one_hour, full_8h / 8.0)

def test_funding_scales_with_notional():
    a = calculate_continuous_funding("BUY", 1.0, 100.0, 0.001, 3600 * 1000)
    b = calculate_continuous_funding("BUY", 2.0, 100.0, 0.001, 3600 * 1000)
    c = calculate_continuous_funding("BUY", 1.0, 200.0, 0.001, 3600 * 1000)
    assert math.isclose(b, 2 * a)
    assert math.isclose(c, 2 * a)

@pytest.mark.parametrize("side", ["BUYY", "SHORT", "", None])
def test_invalid_side_is_rejected(side):
    with pytest.raises(ValueError):
        calculate_continuous_funding(side, 1.0, 100.0, 0.001, 1000)

@pytest.mark.parametrize("qty", [0.0, -1.0])
def test_non_positive_quantity_is_rejected(qty):
    with pytest.raises(ValueError):
        calculate_continuous_funding("BUY", qty, 100.0, 0.001, 1000)

@pytest.mark.parametrize("price", [0.0, -100.0])
def test_non_positive_price_is_rejected(price):
    with pytest.raises(ValueError):
        calculate_continuous_funding("BUY", 1.0, price, 0.001, 1000)

@pytest.mark.parametrize("elapsed_ms", [-1, -1000])
def test_negative_elapsed_is_rejected(elapsed_ms):
    with pytest.raises(ValueError):
        calculate_continuous_funding("BUY", 1.0, 100.0, 0.001, elapsed_ms)

@pytest.mark.parametrize("bad_val", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_inputs_are_rejected(bad_val):
    with pytest.raises(ValueError):
        calculate_continuous_funding("BUY", bad_val, 100.0, 0.001, 1000)
    with pytest.raises(ValueError):
        calculate_continuous_funding("BUY", 1.0, bad_val, 0.001, 1000)
    with pytest.raises(ValueError):
        calculate_continuous_funding("BUY", 1.0, 100.0, bad_val, 1000)

def test_100ms_observations_equal_single_elapsed_interval():
    qty, mark, rate = 1.0, 100.0, 0.001
    total = sum(calculate_continuous_funding("BUY", qty, mark, rate, 100) for _ in range(10))
    expected = calculate_continuous_funding("BUY", qty, mark, rate, 1000)
    assert math.isclose(total, expected, rel_tol=0.0, abs_tol=1e-12)
