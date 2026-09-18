import ast
import unittest
import pandas as pd
import numpy as np

source = open('train_model.py', encoding='utf-8').read()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_anti_leakage')
ns = {'pd': pd, 'np': np, 'log': type('L', (), {'info': lambda *a, **k: None})()}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'train_model.py', 'exec'), ns)
audit_anti_leakage = ns['audit_anti_leakage']

class TestTemporalIntegrity(unittest.TestCase):
    def test_equal_boundary_is_allowed(self):
        df = pd.DataFrame({'open_time':[0], 'close_time':[899999], 'htf1h_source_close_time':[899999], 'htf4h_source_close_time':[899999], 'funding_source_time':[899999]})
        self.assertEqual(audit_anti_leakage(df)['total_violations'], 0)

    def test_future_source_fails(self):
        df = pd.DataFrame({'open_time':[0], 'close_time':[899999], 'htf1h_source_close_time':[900000]})
        with self.assertRaises(ValueError): audit_anti_leakage(df)

    def test_bad_close_before_open_fails(self):
        df = pd.DataFrame({'open_time':[1000], 'close_time':[999]})
        with self.assertRaises(ValueError): audit_anti_leakage(df)

if __name__ == '__main__': unittest.main()
