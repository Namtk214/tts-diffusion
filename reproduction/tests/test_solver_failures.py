import importlib.util
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from benchmark.common import ROOT


class Variable:
    def __init__(self, value):
        self.X = value
    def __rmul__(self, value):
        return 0


class FakeModel:
    def __init__(self, solutions, selected):
        self.Params = SimpleNamespace()
        self.Status, self.SolCount, self.Runtime, self.ObjVal, self.MIPGap = 9, solutions, 5., 1., float("inf")
        self.selected, self.disposed = selected, False
    def addVars(self, n, **kwargs):
        return {i: Variable(int(i in self.selected)) for i in range(n)}
    def addConstr(self, *args, **kwargs):
        pass
    def setObjective(self, *args):
        pass
    def optimize(self):
        pass
    def dispose(self):
        self.disposed = True


def solver(monkeypatch, model):
    monkeypatch.setitem(sys.modules, "gurobipy", SimpleNamespace(Model=lambda: model,
        GRB=SimpleNamespace(BINARY=1, MAXIMIZE=1, TIME_LIMIT=9), quicksum=sum))
    spec = importlib.util.spec_from_file_location("fixture_solver", ROOT / "third_party/sgi/src/my_utils/solvers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gurobi_solver


def test_no_incumbent_fails_without_fallback(monkeypatch):
    model = FakeModel(0, [])
    with pytest.raises(RuntimeError, match="no feasible"):
        solver(monkeypatch, model)(np.ones(8), np.zeros((8, 8)), 4)
    assert model.disposed


def test_time_limited_feasible_incumbent_is_explicit(monkeypatch):
    model = FakeModel(1, [0, 2, 4, 7]); record = {}
    assert solver(monkeypatch, model)(np.ones(8), np.zeros((8, 8)), 4, diagnostics=record) == [0, 2, 4, 7]
    assert record["time_limit_hit"] and record["gap"] is None and model.disposed


def test_wrong_cardinality_is_failure(monkeypatch):
    model = FakeModel(1, [0, 1])
    with pytest.raises(RuntimeError, match="cardinality"):
        solver(monkeypatch, model)(np.ones(8), np.zeros((8, 8)), 4)
