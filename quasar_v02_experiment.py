#!/usr/bin/env python3
"""QUASAR v0.2 pre-registered experiment — progress-driven curriculum.
Author: Claude Fable 5 (Anthropic). Claims V1-V3 + bracket registered before run.
"""
import numpy as np, time
from quasar.quasar import Quasar, Learner, make_holdout

class QuasarV02(Quasar):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._prev_errs = None
    def self_direct_progress(self, bin_errs, temperature=3.0, floor=0.05):
        if self._prev_errs is None:
            self.weights = np.ones(self.n_bins) / self.n_bins
        else:
            prog = np.maximum(self._prev_errs - bin_errs, 1e-9) ** temperature
            w = prog / prog.sum()
            w = np.maximum(w, floor)
            self.weights = w / w.sum()
        self._prev_errs = bin_errs.copy()
    def run_mode(self, mode, rounds, npr, epr, holdout):
        for rd in range(rounds):
            X, Y, _ = self.generate_round(npr)
            for _ in range(epr):
                self.learner.train_step(X, Y)
            be = self.probe_bin_errors()
            if mode == "error":
                self.self_direct(be)
            elif mode == "progress":
                self.self_direct_progress(be)
            # uniform: leave weights alone
        return self.learner.loss(*holdout)


