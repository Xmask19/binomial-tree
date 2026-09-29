# Binomial Option Pricing Model

## Overview

This project implements a binomial options pricing model. The stock price is modelled over discrete time steps, with a fixed probability of moving up or down at each step. The option price is then computed by working backwards from the possible payoffs at maturity.

## Plan

- [x] Stage 0: initialise repository with structure
- [x] Stage 1: Tree with European call
- [x] Stage 2: European put and put-call parity
- [x] Stage 3: Convergence study
- [] Stage 4: American options
- [] Stage 5: Dividends
- [] Stage 6: Trinomial tree


## Convergence

![Convergence to Black-Scholes](output/convergence.png)
The Cox-Ross-Rubinstein (CRR) tree converges to the Black-Scholes price as the number of time steps increases. For the standard test case ($S=100$, $K=100$, $T=1$, $r=0.05$, $\sigma=0.2$), the tree price approaches $10.4506$. 

The convergence is not monotone: the price oscillates around the Black-Scholes value, with the amplitude decreasing at rate $O(1/N)$(Leisen & Reimer, 1996).

## References

Leisen, D. P. J., & Reimer, M. (1996). Binomial models for option valuation — examining and improving convergence. Applied Mathematical Finance, 3(4), 319–346.

## Setup

    pip install -r requirements.txt