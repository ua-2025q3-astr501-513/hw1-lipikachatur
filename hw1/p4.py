#!/usr/bin/env python3
#
# Please look for "TODO" in the comments, which indicate where you
# need to write your code.
#
# Part 4: Solve the Coupled Simple Harmonic Oscillator Problem (1 point)
#
# * Objective:
#   Take the coupled harmonic oscillator problem we solved in class
#   and rewrite it using a well-structured Python class.
# * Details:
#   The description of the problem and the solution template can be
#   found in `hw1/p4.py`.
#
# From lecture `02w`, we solve systems of coupled harmonic oscillators
# semi-analytically by numerically solving eigenvalue problems.
# However, the code structure was not very clean, making the code hard
# to reuse.
# Although numerical analysis in general does not require
# object-oriented programming, it is sometime useful to package
# stateful caluation into classes.
# For this assignment, we will provide a template class.
# Your responsibility to implement the methods in the class.


import numpy as np


class CoupledOscillators:
    """A class to model a system of coupled harmonic oscillators.

    Attributes:
        Omega (np.ndarray): array of angular frequencies of the normal modes.
        V     (np.ndarray): matrix of eigenvectors representing normal modes.
        M0    (np.ndarray): initial amplitudes of the normal modes.

    """

    def __init__(self, X0=[-0.5, 0, 0.5], m=1.0, k=1.0):
        """Initialize the coupled harmonic oscillator system.

        Args:
            X0 (list or np.ndarray): initial displacements of the oscillators.
            m  (float):              mass of each oscillator (assumed identical for all oscillators).
            k  (float):              spring constant (assumed identical for all springs).

        """
        # Construct the stiffness matrix K
        # Solve the eigenvalue problem for K to find normal modes
        # Store angular frequencies and eigenvectors
        # Compute initial modal amplitudes M0 (normal mode decomposition)
        self.m = float(m)
        self.k = float(k)

        # Make X0 a 1D float array; length N defines the system size.
        self.X0 = np.asarray(X0, dtype=float).reshape(-1)
        N = self.X0.size
        if N < 2:
            raise ValueError("Need at least 2 oscillators.")

        # Build stiffness matrix K for fixed ends (wall–masses–wall)
        # K = k * ( 2*I - offdiag(+1) - offdiag(-1) )
        main = 2.0 * np.ones(N)
        off  = -1.0 * np.ones(N - 1)
        K = self.k * (np.diag(main) + np.diag(off, 1) + np.diag(off, -1))

        # Solve eigenproblem for A = K/m (symmetric)
        A = K / self.m
        evals, evecs = np.linalg.eigh(A)   # columns of evecs are modes

        # Angular frequencies and mode shapes
        self.Omega = np.sqrt(np.clip(evals, 0.0, None))
        self.V = evecs

        # Initial modal amplitudes (zero initial velocities)
        # Orthonormal V (since masses identical) => project with V^T
        self.M0 = self.V.T @ self.X0

    
    def __call__(self, t):
        """Calculate the displacements of the oscillators at time t.

        Args:
            t (float): time at which to compute the displacements.

        Returns:
            np.ndarray: displacements of the oscillators at time t.

        """
        # : Reconstruct the displacements from normal modes
        t = float(t)
        q_t = self.M0 * np.cos(self.Omega * t)  # modal coordinates
        return self.V @ q_t

if __name__ == "__main__":

    # Initialize the coupled oscillator system with default parameters
    co = CoupledOscillators()

    # Print displacements of the oscillators at each time step
    print("Time(s)  Displacements")
    print("----------------------")
    for t in np.linspace(0, 10, num=101):
        X = co(t)             # compute displacements at time t
        print(f"{t:.2f}", X)  # print values for reference
