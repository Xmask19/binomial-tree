import pytest

from math import exp

from binomial_tree import (
    binomial_price,
    stock_tree,
    u_d,
    risk_neutral_up_prob,
    time_step
)

S0, K0, T0, R0, SIGMA0 = 100.0, 100.0, 1.0, 0.05, 0.2

BS_CALL = 10.4506
BS_PUT = 5.5735
TREE_N1_CALL = 12.162285

# --- Helpers ---------------------------------------------------------


def test_stock_tree_crr_invariant():
    """
    If the stock price goes up once then down once,
    it should remain the same
    """

    tree = stock_tree(S0, SIGMA0, T0, 4)
    assert tree[2, 1] == pytest.approx(S0)
    assert tree[4, 2] == pytest.approx(S0)


def test_risk_neutral_martingale():
    """Under Q, the expected stock price grows at the risk-free rate."""
    u, d = u_d(SIGMA0, T0, 10)
    p = risk_neutral_up_prob(T0, R0, SIGMA0, 10)
    expected = p * (S0 * u) + (1 - p) * (S0 * d)
    assert expected == pytest.approx(S0 * exp(R0 * time_step(T0, 10)))


def test_probability_in_range():
    from binomial_tree import risk_neutral_up_prob
    p = risk_neutral_up_prob(T0, R0, SIGMA0, 10)
    assert 0 < p < 1


def test_u_d_reciprocal():
    from binomial_tree import u_d
    u, d = u_d(SIGMA0, T0, 4)
    assert u * d == pytest.approx(1.0)

# --- Prices ----------------------------------------------------------


def test_call_n1():
    """
    u = exp(0.2) = 1.2214, d = 1/u = 0.8187
    p = (exp(0.05) - d)/(u - d)) = 0.5775
    up_payoff = 100(u - 1) = 22.1403
    price = exp(-0.05) * p * up_payoff = 12.162285
    """
    price = binomial_price(S0, K0, T0, R0, SIGMA0, 1)
    assert price == pytest.approx(TREE_N1_CALL, abs=1e-6)


def test_call_converges_to_black_scholes():
    """N=1000 should be within 0.01 of the closed form price."""
    price = binomial_price(S0, K0, T0, R0, SIGMA0, 1000)
    assert price == pytest.approx(BS_CALL, abs=0.01)
