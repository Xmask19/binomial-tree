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


def price_on_tree(K: float, sigma: float, r: float, t: float, N: int,
                  tree: np.ndarray, option_type: str, exercise: str,
                  q: float = 0):
    """
    Price an option on a prebuilt stock tree.

    Args and returns match binomial_price, except the tree is
    supplied rather than constructed.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"unknown option type: {option_type}")
    if exercise not in ("european", "american"):
        raise ValueError(f"unknown exercise: {exercise}")
    if option_type == "call":
        payoff = call_payoff
    else:
        payoff = put_payoff
    payoffs = np.zeros((N + 1, N + 1))
    payoffs[N, :] = payoff(tree[N, :], K)
    p = risk_neutral_up_prob(t, r, sigma, N, q)
    disc = step_discount(t, r, N)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            up = payoffs[i + 1, j + 1]
            down = payoffs[i + 1, j]
            continuation = disc * (p * up + (1 - p) * down)
            if exercise == "american":
                intrinsic = payoff(tree[i, j], K)
                payoffs[i, j] = max(continuation, intrinsic)
            else:
                payoffs[i, j] = continuation

    return payoffs[0, 0]


def binomial_price(
    S: float, K: float, t: float, r: float, sigma: float, N: int,
    option_type: str, exercise: str, q: float = 0
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
        q: Continuous dividend yield (annual, decimal). Defaults to 0.

    Returns:
        The option price.
    """

    return price_on_tree(K, sigma, r, t, N, stock_tree(S, sigma, t, N),
                         option_type, exercise, q)


def time_step(T: float, N: int) -> float:
    return T / N


def u_d(sigma: float, T: float, N: int) -> tuple[float, float]:
    u = exp(sigma * sqrt(time_step(T, N)))
    d = 1 / u
    return (u, d)


def risk_neutral_up_prob(t: float, r: float, sigma: float, N: int,
                         q: float = 0) -> float:
    u, d = u_d(sigma, t, N)
    p = (exp((r - q) * time_step(t, N)) - d) / (u - d)
    assert 0 < p < 1, f"p = {p} is not a valid probability"
    return p


def step_discount(t: float, r: float, N: int) -> float:
    return exp(-r * time_step(t, N))


def plot_convergence(S: float = 100, K: float = 100, t: float = 1,
                     r: float = 0.05, sigma: float = 0.2,
                     option_type: str = "call", exercise: str = "european",
                     n=200) -> None:
    """Plot the binomial tree price against number of steps N."""
    # The Black-Scholes reference assumes no dividends. For q > 0,
    # the closed form would need a q-adjustment that this project
    # does not implement.

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
    std = (100, 100, 1, 0.05, 0.2, 1000)
    itm = (150, 100, 1, 0.05, 0.2, 1000)

    print(f"European call: {binomial_price(*std, 'call', 'european'):.4f}")
    print(f"European put:  {binomial_price(*std, 'put', 'european'):.4f}")
    print(f"American call: {binomial_price(*std, 'call', 'american'):.4f}")
    print(f"American put:  {binomial_price(*std, 'put', 'american'):.4f}")

    print()
    print("Deep ITM call, q = 0.10:")
    euro = binomial_price(*itm, "call", "european", q=0.10)
    amer = binomial_price(*itm, "call", "american", q=0.10)
    print(f"  European: {euro:.4f}")
    print(f"  American: {amer:.4f}")

    plot_convergence()
