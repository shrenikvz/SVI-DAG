"""
Exact Markov-equivalence-class enumeration and posterior coverage.

Restored (2026-08-17) as the generator of the table's "Exact rec." column:
results/mec_study/ holds its outputs from the companion linear study
(p=10, s=10, n=1000, 1500 iters, sampling bias -1).

Convention note: SVI-DAG (and the case_3 data generators) store adjacencies
as ``A[i, j] = 1  =>  j -> i`` (column causes row), while
``svidag.utils.dag_to_cpdag`` documents ``dag_adj[i, j] = 1 => i -> j``.
Everything in this module converts to the i->j convention at the boundary,
works there, and converts members back to j->i so they compare bitwise
against posterior samples.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import networkx as nx

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from svidag.utils import dag_to_cpdag  # noqa: E402

#: Brute-force guard.  u reversible edges cost 2^u candidate checks; at
#: sparse ER densities the v-structures compel most edges and u stays small.
MAX_REVERSIBLE = 18


def enumerate_mec(true_adj_j2i: np.ndarray) -> List[np.ndarray]:
    """
    All DAGs Markov-equivalent to ``true_adj_j2i`` (j->i convention, binary).

    CPDAG via svidag's own converter, then brute-force orientation of the
    reversible edges keeping candidates that (a) are acyclic and (b) have the
    same CPDAG.  Check (b) subsumes the "no new v-structures" condition, so
    this is exact -- slower than counting via He & Geng's rooted subclasses,
    but immune to implementation subtleties, which matters more here.
    """
    A = (np.asarray(true_adj_j2i) != 0).astype(int)
    T = A.T.copy()                       # i->j convention
    cpdag = np.asarray(dag_to_cpdag(T)).astype(int)

    und = np.argwhere((cpdag == 1) & (cpdag.T == 1))
    und_pairs = [(int(i), int(j)) for i, j in und if i < j]
    u = len(und_pairs)
    if u > MAX_REVERSIBLE:
        raise RuntimeError(
            f"{u} reversible edges > MAX_REVERSIBLE={MAX_REVERSIBLE}; "
            "2^u enumeration would be too slow -- raise the guard only "
            "deliberately.")

    compelled = ((cpdag == 1) & (cpdag.T == 0)).astype(int)

    members: List[np.ndarray] = []
    for bits in product((0, 1), repeat=u):
        cand = compelled.copy()
        for (i, j), b in zip(und_pairs, bits):
            if b:
                cand[i, j] = 1
            else:
                cand[j, i] = 1
        G = nx.DiGraph(cand)             # already i->j
        if not nx.is_directed_acyclic_graph(G):
            continue
        if not np.array_equal(np.asarray(dag_to_cpdag(cand)).astype(int), cpdag):
            continue
        members.append(cand.T.astype(np.int8))   # back to j->i

    true_key = A.astype(np.int8).tobytes()
    member_keys = {mm.tobytes() for mm in members}
    if true_key not in member_keys:
        raise AssertionError("true DAG missing from its enumerated MEC")
    return members


def mec_coverage(samples_j2i: np.ndarray, true_adj_j2i: np.ndarray) -> Dict[str, float]:
    """
    Coverage of the exact MEC by hard posterior samples.

    ``samples_j2i``: [S, p, p] binary DAG draws (j->i).  Returns:
        mec_size       |MEC|
        mec_cov        fraction of MEC members appearing in >= 1 sample
        mass_in_mec    fraction of samples that are MEC members
        p_true_dag     fraction of samples equal to the true DAG

    The table's "Exact rec." column derives from mec_cov: a seed counts as an
    exact recovery iff mec_cov > 0.
    """
    members = enumerate_mec(true_adj_j2i)
    member_keys = {m.tobytes() for m in members}
    true_key = (np.asarray(true_adj_j2i) != 0).astype(np.int8).tobytes()

    S = samples_j2i.shape[0]
    sample_keys = [
        (np.asarray(samples_j2i[s]) != 0).astype(np.int8).tobytes()
        for s in range(S)
    ]
    seen = set(sample_keys)
    covered = sum(1 for k in member_keys if k in seen)
    in_mec = sum(1 for k in sample_keys if k in member_keys)
    n_true = sum(1 for k in sample_keys if k == true_key)

    return {
        "mec_size": float(len(members)),
        "mec_cov": covered / len(members),
        "mass_in_mec": in_mec / S,
        "p_true_dag": n_true / S,
    }


if __name__ == "__main__":
    chain_i2j = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    members = enumerate_mec(chain_i2j.T)
    assert len(members) == 3, f"chain MEC should have 3 members, got {len(members)}"
    collider_i2j = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    members = enumerate_mec(collider_i2j.T)
    assert len(members) == 1, f"collider MEC should be singleton, got {len(members)}"
    cov = mec_coverage(np.stack([chain_i2j.T, chain_i2j.T]), chain_i2j.T)
    assert cov["mec_size"] == 3 and abs(cov["mec_cov"] - 1 / 3) < 1e-12
    assert cov["mass_in_mec"] == 1.0 and cov["p_true_dag"] == 1.0
    print("mec.py self-test OK")
