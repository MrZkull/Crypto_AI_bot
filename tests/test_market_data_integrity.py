import unittest
import pandas as pd
from market_data_integrity import sanitize_closed_candles, merge_completed_htf
from train_model import make_targets, _add_extra_features, temporal_symbol_split
import train_meta_model as tm


class IntegrityTests(unittest.TestCase):

    def test_forming_candle_rejected_by_observation(self):
        raw = [[0, 1, 2, 0.5, 1.5, 10, 899999]]
        rows = sanitize_closed_candles(raw, observation_time_ms=899998)
        self.assertEqual(rows, [])
        rows_equal = sanitize_closed_candles(raw, observation_time_ms=899999)
        self.assertEqual(len(rows_equal), 1)

    def test_invalid_timestamp_order_rejected_unconditionally(self):
        raw = [[1000, 1, 2, 0.5, 1.5, 10, 999]]
        self.assertEqual(sanitize_closed_candles(raw), [])

    def test_invalid_ohlc_rejected(self):
        raw = [[0, 1, 0.5, 0.8, 1, 10, 899999]]
        self.assertEqual(sanitize_closed_candles(raw), [])

    def test_htf_close_time_alignment(self):
        ltf = pd.DataFrame({'open_time': [45, 60], 'close_time': [58, 74], 'close': [1, 1]})
        htf = pd.DataFrame({'open_time': [0, 60], 'close_time': [59, 119], 'rsi': [10, 20]})
        out = merge_completed_htf(ltf, htf, ['rsi'], 'htf')
        self.assertTrue(pd.isna(out.loc[0, 'htf_rsi']))
        self.assertEqual(out.loc[1, 'htf_rsi'], 10)

    def test_same_candle_tp_sl_is_ambiguous(self):
        df = pd.DataFrame({
            'close': [100] * 26, 'high': [100, 110] + [100] * 24, 'low': [100, 90] + [100] * 24,
            'atr': [1.0] * 26
        })
        y = make_targets(df)
        self.assertEqual(y.iloc[0], 'AMBIGUOUS')

    def test_btc_extra_features_exist_and_are_finite(self):
        n = 40
        df = pd.DataFrame({
            'close': [100 + i * 0.2 for i in range(n)],
            'btc_close': [20000 + i * 5 for i in range(n)],
        })
        out = _add_extra_features(df)
        for col in ('btc_corr_20', 'btc_beta_20', 'btc_rel_strength'):
            self.assertIn(col, out.columns)
            self.assertTrue(out[col].map(pd.notna).all())

    def test_temporal_split_never_reverses_time(self):
        rows = []
        for symbol in ('AAA', 'BBB'):
            for i in range(100):
                rows.append({'symbol': symbol, 'open_time': i, 'regime': 'r1' if i < 40 else 'r2'})
        ds = pd.DataFrame(rows)
        tr, ca, te = temporal_symbol_split(ds, 0.2, 0.15, 2)
        for symbol in ('AAA', 'BBB'):
            tr_s, ca_s, te_s = tr[tr.symbol == symbol], ca[ca.symbol == symbol], te[te.symbol == symbol]
            self.assertLess(tr_s.open_time.max(), ca_s.open_time.min())
            self.assertLess(ca_s.open_time.max(), te_s.open_time.min())

    def test_meta_pipeline_uses_directional_context_features(self):
        df = pd.DataFrame({
            'primary_side': ['BUY', 'SELL'],
            'primary_conf': [0.7, 0.8],
            'rsi': [75, 25],
            'macd_hist': [1, -1],
            'trend': [0.5, -0.5]
        })
        out = tm.augment_meta_features(df)
        self.assertEqual(out['meta_primary_side_code'].tolist(), [1.0, -1.0])
        self.assertEqual(out['meta_rsi_directional'].tolist(), [25.0, 25.0])
        # Non-tautological output schema validation
        self.assertTrue(set(tm.META_SYSTEM_FEATURES).issubset(set(out.columns)))


if __name__ == '__main__':
    unittest.main()
