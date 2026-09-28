# Binomial Option Pricing Model

## Overview

This project implements a binomial options pricing model. The stock price is modelled over discrete time steps, with a fixed probability of moving up or down at each step. The option price is then computed by working backwards from the possible payoffs at maturity.


## Plan

- [x] Stage 0: initialise repository with structure
- [x] Stage 1: Tree with European call
- [] Stage 2: European put and put-call parity
- [] Stage 3: Convergence study
- [] Stage 4: American options
- [] Stage 5: Dividends
- [] Stage 6: Trinomial tree

## Setup

    pip install -r requirements.txt