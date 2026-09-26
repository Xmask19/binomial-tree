"""
Binomial options pricing model.

Implements the binomial options pricing model.
"""

from math import exp, sqrt


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
    raise NotImplementedError("Not yet implemented")


def time_step(T: float, N: int) -> float:
    return T / N


def u_d(sigma: float, T: float, N: int) -> tuple[float, float]:
    u = exp(sigma * sqrt(time_step(T, N)))
    d = 1 / u
    return (u, d)


if __name__ == "__main__":
    print(f"dt for T=1, N=4: {time_step(1, 4)}")
    u, d = u_d(0.2, 1, 4)
    print(f"u = {u:.4f}, d = {d:.4f}")
    print(f"u * d = {u * d:.4f}  (should be 1.0)")