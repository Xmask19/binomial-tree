"""
Binomial options pricing model.

Implements the binomial options pricing model.
"""

import numpy as np
from math import exp, sqrt


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


def backwards_tree(K: float, sigma: float, r: float,
                   t: float, N: int, tree: np.ndarray):
    payoffs = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        terminal_stock = tree[N, j]
        payoffs[N, j] = np.maximum(terminal_stock - K, 0)

    p = risk_neutral_up_prob(t, r, sigma, N)
    disc = step_discount(t, r, N)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            payoffs[i, j] = disc * (p * payoffs[i + 1, j + 1] + (1 - p) * payoffs[i + 1, j])

    return payoffs


def binomial_price(
    S: float, K: float, t: float, r: float, sigma: float, N: int,
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

    return backwards_tree(K, sigma, r, t, N, stock_tree(S, sigma, t, N))[0, 0]


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
    print(f"dt for T=1, N=4: {time_step(1, 4)}")
    u, d = u_d(0.2, 1, 4)
    print(f"u = {u:.4f}, d = {d:.4f}")
    print(f"u * d = {u * d:.4f}  (should be 1.0)")

    print(stock_tree(100, 0.2, 1, 3))

    print(backwards_tree(100, 0.2, 0.05, 1, 3, stock_tree(100, 0.2, 1, 3)))

    for N in [1, 2, 3, 5, 10, 50, 100, 500, 1000]:
        tree = stock_tree(100, 0.2, 1, N)
        price = backwards_tree(100, 0.2, 0.05, 1, N, tree)[0, 0]
        print(f"N={N:4d}  price={price:.4f}  error={price - 10.4506:+.4f}")
