import sys, unittest
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT))
from market_data_integrity import sanitize_closed_candles, sanitize_ohlcv_frame, merge_completed_htf, interval_ms

class TestMarketDataIntegrity(unittest.TestCase):
    def test_all_supported_intervals(self):
        for x in ('1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d'):
            self.assertGreater(interval_ms(x),0)
    def test_duplicate_and_invalid_rows_are_rejected(self):
        rows=[
            {'open_time':0,'close_time':899999,'open':100,'high':110,'low':90,'close':105,'volume':1},
            {'open_time':0,'close_time':899999,'open':100,'high':110,'low':90,'close':105,'volume':1},
            {'open_time':900000,'close_time':1799999,'open':100,'high':90,'low':95,'close':105,'volume':1},
            {'open_time':1800000,'close_time':2699999,'open':100,'high':110,'low':90,'close':105,'volume':1},
        ]
        out=sanitize_closed_candles(rows,candle_duration_ms=900000)
        self.assertEqual(len(out),2)
    def test_raw_array_input(self):
        row=[0,100,110,90,105,1,899999]
        out=sanitize_closed_candles([row],candle_duration_ms=900000)
        self.assertEqual(len(out),1)
    def test_htf_exact_boundary_allowed_and_future_blocked(self):
        base=pd.DataFrame({'open_time':[0,900000,1800000],'close_time':[899999,1799999,2699999]})
        htf=pd.DataFrame({'close_time':[899999,3599999],'rsi':[50,60]})
        out=merge_completed_htf(base,htf,['rsi'],prefix='htf')
        self.assertEqual(out.loc[0,'htf_rsi'],50)
        self.assertEqual(out.loc[1,'htf_rsi'],50)
        self.assertEqual(out.loc[2,'htf_rsi'],50)
        self.assertEqual(out.loc[2,'htf_source_close_time'],899999)

if __name__=='__main__': unittest.main()
