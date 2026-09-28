"""
Binomial options pricing model.

Implements the binomial options pricing model.
"""

import numpy as np
from math import exp, sqrt


def call_payoff(S, K):
    return np.maximum(S - K, 0)


def put_payoff(S, K):
    return np.maximum(K - S, 0)


def stock_tree(S: float, sigma: float, t: float, N: int) -> np.ndarray:
    """
    Build the CRR stock price tree.

    Returns an (N+1) x (N+1) lower-triangular array where
    tree[i, j] = S * u^j * d^(i-j), the stock price at step i
    after j up-moves. Entries with j > i are zero.
    """

    u, d = u_d(sigma, t, N)
    tree = np.zeros((N + 1, N + 1))

    for i in range(N + 1):  # loop over time
        for j in range(i + 1):  # loop over states going up
            tree[i, j] = S * u ** j * d ** (i - j)

    return tree


def price_call_of_tree(K: float, sigma: float, r: float,
                       t: float, N: int, tree: np.ndarray):
    payoffs = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        terminal_stock = tree[N, j]
        payoffs[N, j] = call_payoff(terminal_stock, K)

    p = risk_neutral_up_prob(t, r, sigma, N)
    disc = step_discount(t, r, N)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            payoffs[i, j] = disc * (p * payoffs[i + 1, j + 1] + (1 - p) * payoffs[i + 1, j])

    return payoffs[0, 0]


def binomial_price(
    S: float, K: float, t: float, r: float, sigma: float, N: int,
    option_type: str = "call",
    exercise: str = "european",
) -> float:
    """
    Price a European or American option using a CRR binomial tree.

    Args:
        S: Spot price.
        K: Strike price.
        t: Time to expiry, in years.
        r: Risk-free rate (annual, decimal).
        sigma: Volatility (annual, decimal).
        N: Number of time steps.
        option_type: "call" or "put".
        exercise: "european" or "american".

    Returns:
        The option price.

    Note:
        Only European calls are implemented at present. Put and
        American support are added in later stages.
    """

    return price_call_of_tree(K, sigma, r, t, N, stock_tree(S, sigma, t, N))


def time_step(T: float, N: int) -> float:
    return T / N


def u_d(sigma: float, T: float, N: int) -> tuple[float, float]:
    u = exp(sigma * sqrt(time_step(T, N)))
    d = 1 / u
    return (u, d)


def risk_neutral_up_prob(t: float, r: float, sigma: float, N: int) -> float:
    u, d = u_d(sigma, t, N)
    p = (exp(r * time_step(t, N)) - d) / (u - d)
    assert 0 < p < 1, f"p = {p} is not a valid probability"
    return p


def step_discount(t: float, r: float, N: int) -> float:
    return exp(-r * time_step(t, N))


if __name__ == "__main__":

    pass
