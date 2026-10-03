import pytest

from math import exp, log, sqrt
from scipy.stats import norm
from binomial_tree import (
    binomial_price,
    stock_tree,
    u_d,
    risk_neutral_up_prob,
    time_step,
    bs_call_price
)

S0, K0, T0, R0, SIGMA0, Q0 = 100.0, 100.0, 1.0, 0.05, 0.2, 0.03

BS_CALL = 10.4506
BS_PUT = 5.5735
TREE_N1_CALL = 12.162285
TREE_N1_PUT = 7.285227

d1 = (log(S0 / K0) + (R0 - Q0 + SIGMA0 ** 2 / 2) * T0) / (SIGMA0 * sqrt(T0))
d2 = d1 - SIGMA0 * sqrt(T0)
BS_CALL_DIVIDEND = (
    S0 * exp(- Q0 * T0) * norm.cdf(d1)
    - K0 * exp(-R0 * T0) * norm.cdf(d2))
BS_PUT_DIVIDEND = (
    K0 * exp(-R0 * T0) * norm.cdf(-d2)
    - S0 * exp(-Q0 * T0) * norm.cdf(-d1)
)
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
    N = 10
    u, d = u_d(SIGMA0, T0, N)
    p = risk_neutral_up_prob(T0, R0, SIGMA0, N)
    expected = p * (S0 * u) + (1 - p) * (S0 * d)
    assert expected == pytest.approx(S0 * exp(R0 * time_step(T0, N)))


def test_risk_neutral_growth_with_dividend():
    N = 10
    u, d = u_d(SIGMA0, T0, N)
    p = risk_neutral_up_prob(T0, R0, SIGMA0, N, Q0)
    expected = p * S0 * u + (1 - p) * S0 * d
    assert expected == pytest.approx(S0 * exp((R0 - Q0) * time_step(T0, N)))


def test_probability_in_range():
    p = risk_neutral_up_prob(T0, R0, SIGMA0, 10)
    assert 0 < p < 1


def test_u_d_reciprocal():
    u, d = u_d(SIGMA0, T0, 4)
    assert u * d == pytest.approx(1.0)


def test_bs_call_price_matches_known_value():
    assert bs_call_price(S0, K0, T0, R0, SIGMA0) == pytest.approx(BS_CALL,
                                                                  abs=1e-3)


# --- Prices ----------------------------------------------------------


def test_call_n1():
    """
    u = exp(0.2) = 1.2214, d = 1/u = 0.8187
    p = (exp(0.05) - d)/(u - d)) = 0.5775
    up_payoff = 100(u - 1) = 22.1403
    price = exp(-0.05) * p * up_payoff = 12.162285
    """
    price = binomial_price(S0, K0, T0, R0, SIGMA0, 1, "call", "european")
    assert price == pytest.approx(TREE_N1_CALL, abs=1e-6)


def test_put_n1():
    price = binomial_price(S0, K0, T0, R0, SIGMA0, 1, "put", "european")
    assert price == pytest.approx(TREE_N1_PUT, abs=1e-6)


def test_call_converges_to_black_scholes():
    """N=1000 should be within 0.01 of the closed form price."""
    price = binomial_price(S0, K0, T0, R0, SIGMA0, 1000, "call", "european")
    assert price == pytest.approx(BS_CALL, abs=0.01)


def test_put_converges_to_black_scholes():
    price = binomial_price(S0, K0, T0, R0, SIGMA0, 1000, "put", "european")
    assert price == pytest.approx(BS_PUT, abs=0.01)


def test_call_converges_to_black_scholes_with_dividends():
    price = binomial_price(
        S0, K0, T0, R0, SIGMA0, 1000, "call", "european", Q0)
    assert price == pytest.approx(BS_CALL_DIVIDEND, abs=0.01)


def test_put_converges_to_black_scholes_with_dividends():
    price = binomial_price(
        S0, K0, T0, R0, SIGMA0, 1000, "put", "european", Q0)
    assert price == pytest.approx(BS_PUT_DIVIDEND, abs=0.01)


def test_put_call_parity():
    """Put-call parity should hold for European options."""
    for N in [10, 25, 50, 100]:
        call = binomial_price(S0, K0, T0, R0, SIGMA0, N, "call", "european")
        put = binomial_price(S0, K0, T0, R0, SIGMA0, N, "put", "european")
        assert call - put == pytest.approx(S0 - K0 * exp(-R0 * T0), abs=1e-10)


def test_put_call_parity_with_dividends():
    for N in [10, 25, 50, 100]:
        call = binomial_price(
            S0, K0, T0, R0, SIGMA0, N,
            "call", "european", Q0)
        put = binomial_price(
            S0, K0, T0, R0, SIGMA0, N,
            "put", "european", Q0)
        expected = S0 * exp(-Q0 * T0) - K0 * exp(-R0 * T0)
        assert call - put == pytest.approx(expected, abs=1e-10)


def test_american_call_equals_european_call_without_dividends():
    """
    For a non-dividend-paying stock, an American call has
    the same value as the corresponding European call.
    """
    american = binomial_price(S0, K0, T0, R0, SIGMA0, 100, option_type="call",
                              exercise="american")
    european = binomial_price(S0, K0, T0, R0, SIGMA0, 100, option_type="call",
                              exercise="european")
    assert american == pytest.approx(european, abs=1e-10)


def test_american_put_greater():
    """
    Early exercise makes an American put
    more valuable than a European one.
    """
    american = binomial_price(S0, K0, T0, R0, SIGMA0, 100, option_type="put",
                              exercise="american")
    european = binomial_price(S0, K0, T0, R0, SIGMA0, 100, option_type="put",
                              exercise="european")
    assert american > european


def test_dividends_reduce_call_prices():
    no_div = binomial_price(S0, K0, T0, R0, SIGMA0, 1000, "call", "european")
    with_div = binomial_price(S0, K0, T0, R0, SIGMA0, 1000, "call", "european",
                              q=0.05)
    assert with_div < no_div


def test_dividends_increase_put_prices():
    no_div = binomial_price(S0, K0, T0, R0, SIGMA0, 1000, "put", "european")
    with_div = binomial_price(S0, K0, T0, R0, SIGMA0, 1000,
                              "put", "european", q=0.05)
    assert with_div > no_div


def test_dividend_can_make_american_call_more_valuable():
    S = 150.0
    K = 100.0
    q = 0.10

    european = binomial_price(
        S, K, T0, R0, SIGMA0, 500,
        "call", "european", q,
    )
    american = binomial_price(
        S, K, T0, R0, SIGMA0, 500,
        "call", "american", q,
    )

    assert american > european
