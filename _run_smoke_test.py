"""Standalone smoke test runner - patches ortools before any project imports."""
import sys, os, types

# --- Pre-mock ortools BEFORE loading any aps_cp_sat code ---
_mock_ortools = types.ModuleType('ortools')
_mock_cp = types.ModuleType('ortools.sat')
_mock_cp_model = types.ModuleType('ortools.sat.python')
sys.modules['ortools'] = _mock_ortools
sys.modules['ortools.sat'] = _mock_cp
sys.modules['ortools.sat.python'] = _mock_cp_model

class _MockIntVar:
    def __init__(self, name='', lb=0, ub=0): self._name = name
    def solution_value(self): return 0
    def __add__(self, other): return self
    def __radd__(self, other): return self
    def __sub__(self, other): return self
    def __rsub__(self, other): return self
    def __mul__(self, other): return self
    def __rmul__(self, other): return self
    def __eq__(self, other): return False
    def __le__(self, other): return True
    def __ge__(self, other): return True
    def __lt__(self, other): return False
    def __gt__(self, other): return False
    def __neg__(self): return self
    def __hash__(self): return hash(self._name)

class _MockCpModel:
    def __init__(self): self._bvars, self._ivars, self._constraints = {}, {}, []
    def NewBoolVar(self, name): v = _MockIntVar(name); self._bvars[name] = v; return v
    def NewIntVar(self, lb, ub, name): v = _MockIntVar(name, lb, ub); self._ivars[name] = v; return v
    def Add(self, expr=None):
        self._constraints.append(expr)
        class _FC:
            def __and__(self, other): return self
            def OnlyEnforceIf(self, v): return self
            def OnlyEnforceIfNot(self, v): return self
        return _FC()
    def AddCircuit(self, arcs): self._constraints.append(('circuit', arcs))
    def AddHint(self, var, val): pass
    def AddMultiObjectiveRegression(self, *a, **kw): pass
    def Maximize(self, var): pass
    def Minimize(self, var): pass

class _MockCpSolver:
    def __init__(self):
        class _MSP:
            log_search_progress = False; num_workers = 1; random_seed = 42; max_time_in_seconds = 5.0
        self.parameters = _MSP(); self._status = 1
    def Solve(self, model): return 1
    def Value(self, var): return 1
    def StatusName(self, s): return {1:'OPTIMAL',2:'FEASIBLE',3:'INFEASIBLE'}.get(s,'UNKNOWN')

_mock_cp_model.cp_model = _MockCpModel
_mock_cp_model.CpModel = _MockCpModel
_mock_cp_model.IntVar = _MockIntVar
_mock_cp_model.CpSolver = _MockCpSolver
_mock_cp_model.OPTIMAL = 1; _mock_cp_model.FEASIBLE = 2; _mock_cp_model.INFEASIBLE = 3
_mock_cp_model.UNBOUNDED = 4; _mock_cp_model.MODEL_INVALID = 5; _mock_cp_model.UNKNOWN = 6

# --- Mock openpyxl ---
def _mock_ox(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

_mock_ox('openpyxl')
_openpyxl_utils = _mock_ox('openpyxl.utils')
_openpyxl_utils.get_column_letter = lambda idx: str(idx)
_ox = sys.modules['openpyxl']
_ox.utils = _openpyxl_utils
_ox_wb = _mock_ox('openpyxl.workbook')
_ox_wb.Workbook = type('W', (), {'active': None})
_ox_st = _mock_ox('openpyxl.styles')
_ox_st.Font = type('Font', (), {})
_ox_st.Alignment = type('Align', (), {})
_ox_st.PatternFill = type('Fill', (), {})
_ox_st.Border = type('Border', (), {})
_ox_st.Side = type('Side', (), {})

# --- Read test file and patch the ortools try/except block ---
sys.path.insert(0, 'src')

test_path = 'D:/Develop/WorkSpace/APS/src/aps_cp_sat/model/tests_smoke_constructive_lns.py'
with open(test_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()  # Keep raw lines (with newlines)

# Locate the ortools try/except block: lines[18]='try:\n', line[19]=from, line[20]='except ModuleNotFoundError:'
# After that, skip all indented lines until we see a non-indented line
SKIP_START = None
SKIP_END = None

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == 'try:' and i == 18:  # line 19 (0-indexed 18)
        SKIP_START = i
    if SKIP_START is not None and stripped.startswith('except ModuleNotFoundError:'):
        # Find the first non-blank, non-indented line after the except body
        except_indent = len(line) - len(line.lstrip())
        SKIP_END = i + 1  # skip the 'except' line
        # Skip all lines that start with indent > except_indent
        j = SKIP_END
        while j < len(lines):
            l = lines[j]
            if l.strip() == '' or l.strip().startswith('#'):
                j += 1
                continue
            indent = len(l) - len(l.lstrip())
            if indent > except_indent:
                j += 1
                continue
            else:
                break
        # Skip all lines from SKIP_START to j-1
        SKIP_END = j
        break

patched = []
for i, line in enumerate(lines):
    if SKIP_START is not None and SKIP_START <= i < SKIP_END:
        if i == SKIP_START:
            patched.append('import sys; cp_model = sys.modules["ortools.sat.python"].cp_model  # pre-mocked at runtime\n')
    else:
        patched.append(line)

content = ''.join(patched)

import tempfile, importlib.util
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', dir='D:/Develop/WorkSpace/APS') as tf:
    tf.write(content)
    tmp_path = tf.name

try:
    spec = importlib.util.spec_from_file_location('_smoke', tmp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exit_code = mod._main()
finally:
    try: os.unlink(tmp_path)
    except: pass

sys.exit(exit_code)