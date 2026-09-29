"""
Binomial options pricing model.

Implements the binomial options pricing model.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp


def bs_call_price(S, K, t, r, sigma):
    d1 = (log(S / K) + (r + sigma**2 / 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    return S * norm.cdf(d1) - K * exp(-r * t) * norm.cdf(d2)


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


def price_european_of_tree(K: float, sigma: float, r: float, t: float, N: int,
                           tree: np.ndarray, option_type: str):
    payoffs = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        terminal_stock = tree[N, j]
        if option_type == "call":
            payoffs[N, j] = call_payoff(terminal_stock, K)
        elif option_type == "put":
            payoffs[N, j] = put_payoff(terminal_stock, K)
        else:
            raise ValueError("Not a valid option type, expected call or put")

    p = risk_neutral_up_prob(t, r, sigma, N)
    disc = step_discount(t, r, N)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            up = payoffs[i+1, j+1]
            down = payoffs[i+1, j]

            payoffs[i, j] = disc * (p * up + (1 - p) * down)

    return payoffs[0, 0]


def binomial_price(
    S: float, K: float, t: float, r: float, sigma: float, N: int,
    option_type: str,
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
        Only European options are implemented, American options will be added.
    """

    return price_european_of_tree(K, sigma, r, t, N,
                                  stock_tree(S, sigma, t, N), option_type)


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


def plot_convergence(S: float = 100, K: float = 100, t: float = 1,
                     r: float = 0.05, sigma: float = 0.2,
                     option_type: str = "call", exercise: str = "european",
                     n=200) -> None:

    bs_price = bs_call_price(S, K, t, r, sigma)
    prices = np.array([binomial_price(S, K, t, r, sigma, N, option_type,
                                      exercise) for N in range(1, n + 1)])

    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(1, n + 1),
        prices,
        linewidth=1,
        label="Binomial Price",
    )
    plt.axhline(bs_price, color="red", linestyle="--",
                label=f"Black-Scholes ({bs_price:.2f})")
    plt.xlabel("Number of steps N")
    plt.ylabel("European call price")
    plt.title("Binomial tree convergence to Black-Scholes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("output/convergence.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_convergence()
