from networkx import DiGraph, draw
from dowhy import gcm
from numpy import mean, var
import numpy as np
import concurrent.futures
from typing import Iterable, List, Tuple, Dict, Any, Optional
from threading import Lock


# SCM and do-calculus operations
class SCM():
    """
    Light wrapper around a networkx DiGraph + DoWhy-GCM model.
    Adds:
      - internal caches for do(x) -> (E, V)
      - batched evaluation helpers
    """
    def __init__(self, graph: DiGraph | list, set_output: Optional[str] = None, structural_equations: Optional[Dict[str, Any]] = None):
        if isinstance(graph, DiGraph):
            self.graph = graph
        else:
            try:
                self.graph = DiGraph(graph)
            except Exception as e:
                raise Exception('Graph must be networkx.DiGraph object or networkx.DiGraph formatted list.') from e

        sinks = [n for n, d in self.graph.out_degree() if d == 0]

        if set_output is not None:
            self.output_node = set_output
        else:
            if len(sinks) == 0:
                raise ValueError("Graph has no sink (output) node; specify `set_output`.")
            if len(sinks) > 1:
                # be explicit to avoid silent wrong picks
                print(f"Graph has multiple sink nodes {sinks}; specify `set_output`. Using the first sink: {sinks[0]}") 
            self.output_node = sinks[0]

        self.causal_model = gcm.StructuralCausalModel(self.graph)

        if structural_equations is not None:
            for var in self.graph.nodes:
                if var not in structural_equations:
                    # For root nodes (no parents), allow missing mechanism
                    if not list(self.graph.predecessors(var)):
                        continue
                    raise ValueError(f"Missing structural equation for node '{var}'")
                self.causal_model.set_causal_mechanism(
                    var,
                    gcm.causal_mechanisms.FunctionCausalMechanism(structural_equations[var])
                )

        # In-memory cache: key=(tuple(vars), tuple(vals)) -> (mean, var)
        self._do_cache: Dict[Tuple[Tuple[str, ...], Tuple[float, ...]], Tuple[float, float]] = {}

        # Keep the last fitted data around for interventional sampling conditioning.
        self.observational_samples = None

    def fit(self, observational_samples, init: bool = False):
        self.observational_samples = observational_samples
        if init:
            try:
                _ = self.causal_model.causal_mechanism(self.output_node)
                # mechanisms already set
            except Exception:
                # auto-assign mechanisms
                gcm.auto.assign_causal_mechanisms(self.causal_model, observational_samples)
        gcm.fit(self.causal_model, observational_samples)
        # fitting changes the model, cached do-statistics may be stale
        self._do_cache.clear()

    # Draw graph
    def draw(self):
        draw(self.graph, with_labels=True)

    def _do_key(self, interventional_variable: Iterable[str], interventional_value: Iterable[float]) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
        return (tuple(interventional_variable), tuple(float(v) for v in interventional_value))

    def _compute_do_stats_single(
        self,
        interventional_variable: List[str],
        interventional_value: List[float],
        n_samples: int = 200
    ) -> Tuple[float, float]:
        """
        Compute (mean, variance) of the output under do(X = value).
        Cached by (vars, vals).
        """
        key = self._do_key(interventional_variable, interventional_value)
        if key in self._do_cache:
            return self._do_cache[key]

        # Build intervention dict of constant functions
        intervention_dict = {k: (lambda v: (lambda _: v))(v)
                             for k, v in zip(interventional_variable, interventional_value)}

        if self.observational_samples is not None:
            # Conditional sampling (observed_data implies we can’t set num_samples_to_draw)
            samples = gcm.interventional_samples(
                self.causal_model,
                intervention_dict,
                observed_data=self.observational_samples
                # num_samples_to_draw=n_samples
            )
        else:
            # Unconditional sampling
            samples = gcm.interventional_samples(
                self.causal_model,
                intervention_dict,
                num_samples_to_draw=n_samples
            )

        y = samples[self.output_node]
        m = float(mean(y))
        v = float(var(y))

        if key not in self._do_cache:
            self._do_cache[key] = (m, v)

        return m, v

    def do_mean_var_batch(
        self,
        interventional_variable: List[str],
        interventional_values: np.ndarray | List[List[float]],
        n_samples: int = 200,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batched (means, variances) for many do-values.
          interventional_values: shape (N, d)
        Uses cache for previously computed points, and computes missing ones
        (optionally) in parallel.
        """
        vals = np.asarray(interventional_values, dtype=float)
        if vals.ndim == 1:
            vals = vals[None, :]
        N = vals.shape[0]

        means = np.empty(N, dtype=float)
        vars_ = np.empty(N, dtype=float)

        # Figure out which are cached
        keys = [self._do_key(interventional_variable, vals[i, :]) for i in range(N)]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._do_cache]

        # Compute missing
        if missing_idx:
            def _work(i):
                _, v = interventional_variable, vals[i, :].tolist()
                return i, self._compute_do_stats_single(interventional_variable, v, n_samples)

            if parallel and len(missing_idx) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for i, (m, v) in ex.map(_work, missing_idx):
                        # cache is updated inside _compute_do_stats_single already
                        pass
            else:
                for i in missing_idx:
                    self._compute_do_stats_single(interventional_variable, vals[i, :].tolist(), n_samples)

        # Read out (now all in cache)
        for i, k in enumerate(keys):
            m, v = self._do_cache[k]
            means[i] = m
            vars_[i] = v

        return means, vars_


# Simple wrappers (backwards compatible)
def E_output_given_do(interventional_variable: list[str], interventional_value: list[float], causal_model: SCM, n_samples: int = 200):
    m, _ = causal_model._compute_do_stats_single(interventional_variable, interventional_value, n_samples=n_samples)
    return m

def V_output_given_do(interventional_variable: list[str], interventional_value: list[float], causal_model: SCM, n_samples: int = 200):
    _, v = causal_model._compute_do_stats_single(interventional_variable, interventional_value, n_samples=n_samples)
    return v
