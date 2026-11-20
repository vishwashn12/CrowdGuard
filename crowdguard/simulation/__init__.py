"""Closed-loop evaluation of congestion-control policies.

A macroscopic crowd model of a multi-gate venue, the observation path that
turns its true state into the ``CrowdState`` objects the live system feeds the
decision engine, interchangeable control policies, and a Monte Carlo runner.

Like ``crowdguard.decision`` this package imports no vision or model code, so
the engine is evaluated exactly as deployed, without cameras.
"""
