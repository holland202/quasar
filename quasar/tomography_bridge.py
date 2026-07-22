"""quasar/tomography_bridge.py

Drop-in bridge: wraps the QUASAR closed loop with finite-shot tomography.
No modifications to quasar.py needed.

Usage:
    from quasar.tomography_bridge import TomographicQuasarGenerator
    gen = TomographicQuasarGenerator(
        base_generator=your_generator,
        use_tomography=True,
        shots=512,
        n_bases=6,
        method='mle',
    )
    result = gen.generate_trajectory(n_steps=10)
    states = result['states']  # reconstructed Bloch vectors when tomography is on
"""

import numpy as np
from quasar.finite_shot_tomography import (
    TomographicTrajectoryGenerator,
    bloch_to_rho,
    rho_to_bloch,
    bures_distance,
)


class TomographicQuasarGenerator:
    """
    Wraps any base generator to optionally produce tomographic reconstructions.
    
    Interface compatible with quasar.QuasarGenerator:
        - generate_trajectory(n_steps) -> dict with 'states' key
        - generate_batch(n, n_steps) -> list of trajectories
    
    When use_tomography=False, passes through to base generator unchanged.
    When use_tomography=True, returns reconstructed Bloch vectors as 'states'.
    """

    def __init__(self, base_generator, use_tomography=False,
                 shots=512, n_bases=6, method='mle', seed=None):
        self.base_generator = base_generator
        self.use_tomography = use_tomography
        
        if use_tomography:
            self.tomogen = TomographicTrajectoryGenerator(
                base_generator=base_generator,
                shots=shots,
                n_bases=n_bases,
                reconstructor_method=method,
                seed=seed,
            )
            self.shots = shots
            self.n_bases = n_bases
        else:
            self.tomogen = None

    def generate_trajectory(self, n_steps=10):
        """
        Generate a trajectory. Returns dict with:
            - 'states': list of Bloch vectors (reconstructed if tomography on)
            - 'exact_states': list of exact density matrices (always present)
            - 'difficulty': Bures path length
            - 'reconstruction_errors': list of Bures distances (if tomography on)
            - 'fidelities': list of fidelities (if tomography on)
        """
        if not self.use_tomography:
            exact = self.base_generator.generate_trajectory(n_steps)
            exact_states = exact['states']
            # Convert density matrices to Bloch vectors for uniform interface
            states = [rho_to_bloch(rho) for rho in exact_states]
            difficulty = self._compute_difficulty(exact_states)
            return {
                'states': states,
                'exact_states': exact_states,
                'difficulty': difficulty,
            }
        
        # Tomography path
        result = self.tomogen.generate_trajectory(n_steps)
        exact_states = result['exact_states']
        recon_states = result['reconstructed_states']
        
        # Compute difficulty on reconstructed trajectory
        recon_rhos = [bloch_to_rho(r) for r in recon_states]
        difficulty = self._compute_difficulty(recon_rhos)
        
        return {
            'states': recon_states,
            'exact_states': exact_states,
            'difficulty': difficulty,
            'reconstruction_errors': result['reconstruction_errors'],
            'fidelities': result['fidelities'],
            'shots': result['shots'],
            'n_bases': result['n_bases'],
        }

    def generate_batch(self, n_trajectories, n_steps=10):
        """Generate multiple trajectories."""
        return [self.generate_trajectory(n_steps) for _ in range(n_trajectories)]

    @staticmethod
    def _compute_difficulty(states):
        """Compute Bures path length (total Bures distance along trajectory)."""
        total = 0.0
        for i in range(len(states) - 1):
            total += bures_distance(states[i], states[i + 1])
        return total


def demo_closed_loop_with_tomography():
    """
    Self-contained demo: run the closed loop with and without tomography,
    compare difficulty distributions and reconstruction quality.
    """
    print("=" * 60)
    print("TOMOGRAPHY BRIDGE — CLOSED LOOP DEMO")
    print("=" * 60)

    # Minimal base generator for demo
    class DummyGenerator:
        def __init__(self, seed=42):
            self.rng = np.random.default_rng(seed)
        def generate_trajectory(self, n_steps=10):
            r = np.array([0, 0, 1], dtype=float)
            states = [bloch_to_rho(r)]
            for _ in range(n_steps - 1):
                n = self.rng.standard_normal(3)
                n = n / np.linalg.norm(n)
                H = n[0] * np.array([[0, 1], [1, 0]], complex) + \
                    n[1] * np.array([[0, -1j], [1j, 0]], complex) + \
                    n[2] * np.array([[1, 0], [0, -1]], complex)
                U = np.linalg.matrix_power(np.eye(2, dtype=complex) - 1j * H * 0.1, 10)
                U = U / np.linalg.det(U) ** 0.5
                rho = U @ states[-1] @ U.conj().T
                r_vec = rho_to_bloch(rho)
                r_vec = (1 - 0.05) * r_vec  # depolarizing
                states.append(bloch_to_rho(r_vec))
            return {'states': states}

    base = DummyGenerator(seed=42)

    # Exact path
    exact_gen = TomographicQuasarGenerator(base, use_tomography=False)
    exact_traj = exact_gen.generate_trajectory(n_steps=10)
    
    # Tomographic path
    tomogen = TomographicQuasarGenerator(
        base, use_tomography=True,
        shots=512, n_bases=6, method='mle', seed=42
    )
    tom_traj = tomogen.generate_trajectory(n_steps=10)

    print(f"\nExact trajectory:")
    print(f"  Difficulty (Bures path): {exact_traj['difficulty']:.4f}")
    print(f"  States: {len(exact_traj['states'])} Bloch vectors")

    print(f"\nTomographic trajectory (512 shots, 6 bases, MLE):")
    print(f"  Difficulty (Bures path): {tom_traj['difficulty']:.4f}")
    print(f"  Mean fidelity: {np.mean(tom_traj['fidelities']):.4f}")
    print(f"  Mean Bures error: {np.mean(tom_traj['reconstruction_errors']):.4f}")
    print(f"  Difficulty bias: {tom_traj['difficulty'] - exact_traj['difficulty']:.4f}")

    # Batch comparison
    print(f"\nBatch comparison (20 trajectories):")
    exact_batch = exact_gen.generate_batch(20, n_steps=10)
    tom_batch = tomogen.generate_batch(20, n_steps=10)
    
    exact_diffs = [t['difficulty'] for t in exact_batch]
    tom_diffs = [t['difficulty'] for t in tom_batch]
    
    print(f"  Exact difficulty:  mean={np.mean(exact_diffs):.4f}, std={np.std(exact_diffs):.4f}")
    print(f"  Tomo difficulty:   mean={np.mean(tom_diffs):.4f}, std={np.std(tom_diffs):.4f}")
    print(f"  Correlation: {np.corrcoef(exact_diffs, tom_diffs)[0,1]:.4f}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nThe bridge is ready. Use:")
    print("  from quasar.tomography_bridge import TomographicQuasarGenerator")
    print("  gen = TomographicQuasarGenerator(base_gen, use_tomography=True)")


if __name__ == "__main__":
    demo_closed_loop_with_tomography()
