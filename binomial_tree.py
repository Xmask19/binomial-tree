"""
Binomial options pricing model.

Implements the binomial options pricing model.
"""

import numpy as np
from math import exp, sqrt


def stock_tree(S: float, sigma: float, T: float, N: int) -> np.ndarray:
    """
    Build the CRR stock price tree.

    Returns an (N+1) x (N+1) lower-triangular array where
    tree[i, j] = S * u^j * d^(i-j), the stock price at step i
    after j up-moves. Entries with j > i are zero.
    """

    u, d = u_d(sigma, T, N)
    tree = np.zeros((N + 1, N + 1))
    # tree[0, 0] = S

    for i in range(N + 1):  # loop over time
        for j in range(i + 1):  # loop over states going up
            tree[i, j] = S * u ** j * d ** (i - j)

    return tree


def binomial_price(
    S: float, K: float, T: float, r: float, sigma: float, N: int,
    option_type: str = "call",
    exercise: str = "european",
) -> float:
    """
    Price a European or American option using a CRR binomial tree.

    Args
    S : Spot price
    K : Strike price
    T: Time to expiry, in years
    r : risk-free rate
    sigma : volatility
    N : Number of time steps
    option_type : Determines the payoff function at expiry (call or put)
    exercise : Determines whether early exercise is allowed
        (European vs American)"""


    prices_array = []

    raise NotImplementedError("Not yet implemented")


def time_step(T: float, N: int) -> float:
    return T / N


def u_d(sigma: float, T: float, N: int) -> tuple[float, float]:
    u = exp(sigma * sqrt(time_step(T, N)))
    d = 1 / u
    return (u, d)


def risk_neutral_up_prob(
        S: float, K: float, T: float, r: float, sigma: float, N: int) -> float:
    u, d = u_d(sigma, T, N)
    return (exp(r * time_step(T, N)) - d/(u - d))


def step_discount(T: float, r: float, N: int) -> float:
    return exp(-r * time_step(T, N))


if __name__ == "__main__":
    print(f"dt for T=1, N=4: {time_step(1, 4)}")
    u, d = u_d(0.2, 1, 4)
    print(f"u = {u:.4f}, d = {d:.4f}")
    print(f"u * d = {u * d:.4f}  (should be 1.0)")

    print(stock_tree(100, 0.2, 1, 3))
