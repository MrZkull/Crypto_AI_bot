#test_execution_policy_contract

import sys, unittest
from pathlib import Path
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT))
import execution_policy as ep
class TestExecutionPolicyContract(unittest.TestCase):
    def test_identity_functions_exist(self):
        for name in ('get_policy_hash','get_feature_code_hash','get_feature_schema_hash','build_candidate_config','get_config_hash'):
            self.assertTrue(callable(getattr(ep,name)))
    def test_policy_hash_is_real(self):
        self.assertEqual(len(ep.get_policy_hash()),64)
    def test_schema_hash_stable(self):
        self.assertEqual(ep.get_feature_schema_hash(['a','b']),ep.get_feature_schema_hash(['a','b']))
        self.assertNotEqual(ep.get_feature_schema_hash(['a','b']),ep.get_feature_schema_hash(['b','a']))
if __name__=='__main__': unittest.main()
