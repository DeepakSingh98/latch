"""LATCH-MATRIX: existing fast matrix bodies exposed as explicit, resource-checked candidates.

Public API (see HANDOFF.md):
    MatrixProblem, Operand, dense, linear_products, expert_products     (problem.py)
    Requirements, Precision, Synchronization, ProgramResources, check   (requirements.py)
    candidates, select, POLICIES                                        (candidates.py)
    MatrixProvider                                                      (provider.py)

Imports of the installed operation compiler are lazy and need PyTorch.
No research checkout or additional compiler import path is required.
"""
from .problem import MatrixProblem, Operand, dense, linear_products, expert_products
from .requirements import Requirements, Precision, Synchronization, ProgramResources, check
from .candidates import candidates, select, POLICIES, Candidate
from .provider import MatrixProvider

__all__ = ['MatrixProblem', 'Operand', 'dense', 'linear_products', 'expert_products', 'Requirements', 'Precision',
           'Synchronization', 'ProgramResources', 'check', 'candidates', 'select', 'POLICIES', 'Candidate',
           'MatrixProvider']
