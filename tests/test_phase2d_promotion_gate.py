import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from phase2d_promotion_gate import evaluate_promotion
class TestPhase2DPromotionGate(unittest.TestCase):
    def _records(self,n,net_r,status='TP',offset=0):
        return [{'open_time':(offset+i)*15*60*1000,'net_r':net_r,'status':status} for i in range(n)]
    def test_insufficient_is_fail_closed(self):
        r=evaluate_promotion(self._records(49,1.0),self._records(50,.1))
        self.assertFalse(r['promotion_ready'])
        self.assertIn('insufficient sample',r['reason'])
    def test_positive_candidate_not_enough_if_not_better(self):
        r=evaluate_promotion(self._records(60,.20),self._records(60,.30,offset=1000))
        self.assertFalse(r['promotion_ready'])
        self.assertFalse(r['candidate_beats_production'])
    def test_ambiguous_and_invalid_are_excluded(self):
        c=self._records(50,.5)+self._records(20,99.0,status='AMBIGUOUS',offset=1000)
        p=self._records(50,.1,offset=2000)
        r=evaluate_promotion(c,p)
        self.assertEqual(r['n_candidate'],50)
        self.assertTrue(r['promotion_ready'])
if __name__=='__main__': unittest.main()
