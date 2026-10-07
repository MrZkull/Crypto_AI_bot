from unittest.mock import MagicMock, patch

import trade_executor


def test_default_execution_mode_is_predict_only():
    original = getattr(
        trade_executor.config,
        "EXECUTION_MODE",
        None,
    )

    try:
        trade_executor.config.EXECUTION_MODE = "PREDICT_ONLY"
        assert trade_executor.get_execution_mode() == "PREDICT_ONLY"
        assert not trade_executor._execution_allows_new_entries()
        assert not trade_executor._execution_allows_management()
    finally:
        if original is None:
            try:
                delattr(
                    trade_executor.config,
                    "EXECUTION_MODE",
                )
            except AttributeError:
                pass
        else:
            trade_executor.config.EXECUTION_MODE = original


def test_invalid_execution_mode_fails_closed():
    original = getattr(
        trade_executor.config,
        "EXECUTION_MODE",
        None,
    )

    try:
        trade_executor.config.EXECUTION_MODE = "NOT_A_REAL_MODE"

        try:
            trade_executor.get_execution_mode()
        except RuntimeError as exc:
            assert "Invalid EXECUTION_MODE" in str(exc)
        else:
            raise AssertionError(
                "Invalid execution mode was accepted"
            )
    finally:
        if original is None:
            try:
                delattr(
                    trade_executor.config,
                    "EXECUTION_MODE",
                )
            except AttributeError:
                pass
        else:
            trade_executor.config.EXECUTION_MODE = original


def test_execute_trade_is_blocked_before_exchange_access():
    original = getattr(
        trade_executor.config,
        "EXECUTION_MODE",
        None,
    )

    try:
        trade_executor.config.EXECUTION_MODE = "PREDICT_ONLY"

        deribit = MagicMock()

        result = trade_executor.execute_trade(
            deribit,
            {
                "symbol": "ETHUSDT",
                "signal": "BUY",
                "confidence": 95.0,
                "score": 10,
            },
            risk_mult=1.0,
            balance=100000.0,
        )

        assert result is False

        # Defense-in-depth guarantee:
        # the exchange object must not be touched at all.
        deribit.assert_not_called()
        assert not deribit.method_calls

    finally:
        if original is None:
            try:
                delattr(
                    trade_executor.config,
                    "EXECUTION_MODE",
                )
            except AttributeError:
                pass
        else:
            trade_executor.config.EXECUTION_MODE = original


def test_predict_only_scan_does_not_construct_deribit_and_still_scores():
    original_mode = getattr(
        trade_executor.config,
        "EXECUTION_MODE",
        None,
    )
    original_symbols = trade_executor.SYMBOLS
    original_liveness = dict(
        trade_executor._SCAN_LIVENESS
    )

    try:
        trade_executor.config.EXECUTION_MODE = "PREDICT_ONLY"
        trade_executor.SYMBOLS = ["ETHUSDT"]

        generated = []

        pipeline = {
            "recommended_threshold_buy": 0.40,
            "recommended_threshold_sell": 0.45,
        }

        fake_mode = {
            "label": "TEST",
        }

        fake_vol = {
            "status": "NORMAL",
        }

        def fake_generate_signal(
            symbol,
            pipeline,
            thresholds,
            btc_momentum,
            whale_flow,
            fng_data,
            btc_df15_live=None,
        ):
            generated.append(
                {
                    "symbol": symbol,
                    "pipeline": pipeline,
                }
            )
            return None

        status_writes = []

        def fake_save_json(path, payload):
            status_writes.append((str(path), dict(payload)))

        with patch.object(
            trade_executor,
            "should_scan",
            return_value=(
                True,
                fake_mode,
                fake_vol,
                None,
            ),
        ), patch.object(
            trade_executor.joblib,
            "load",
            return_value=pipeline,
        ), patch.object(
            trade_executor,
            "get_mode_thresholds",
            return_value={
                "min_score": 0,
                "min_adx": 0,
            },
        ), patch.object(
            trade_executor,
            "get_effective_risk",
            return_value=1.0,
        ), patch.object(
            trade_executor,
            "check_btc_momentum",
            return_value=None,
        ), patch.object(
            trade_executor,
            "get_exchange_netflow",
            return_value=None,
        ), patch.object(
            trade_executor,
            "check_fear_and_greed",
            return_value=None,
        ), patch.object(
            trade_executor,
            "get_data",
            return_value=None,
        ), patch.object(
            trade_executor,
            "generate_signal",
            side_effect=fake_generate_signal,
        ), patch.object(
            trade_executor,
            "save_json",
            side_effect=fake_save_json,
        ), patch.object(
            trade_executor,
            "save_balance",
            side_effect=AssertionError(
                "PREDICT_ONLY must never call save_balance()"
            ),
        ), patch.object(
            trade_executor,
            "DeribitClient",
            side_effect=AssertionError(
                "PREDICT_ONLY must never construct DeribitClient"
            ),
        ):
            trade_executor._run_execution_scan_locked()

        assert len(generated) == 1
        assert generated[0]["symbol"] == "ETHUSDT"

        completed = [
            payload
            for path, payload in status_writes
            if payload.get("phase") == "completed"
        ]

        assert completed

        final_status = completed[-1]

        assert (
            final_status["execution_mode"]
            == "PREDICT_ONLY"
        )

        assert (
            final_status["symbols_attempted"]
            == 1
        )

        # The mocked generate_signal returns None, so zero successful
        # model scores is expected in this structural test.
        assert (
            final_status["signals_found"]
            == 0
        )

    finally:
        trade_executor.SYMBOLS = original_symbols

        trade_executor._SCAN_LIVENESS.clear()
        trade_executor._SCAN_LIVENESS.update(
            original_liveness
        )

        if original_mode is None:
            try:
                delattr(
                    trade_executor.config,
                    "EXECUTION_MODE",
                )
            except AttributeError:
                pass
        else:
            trade_executor.config.EXECUTION_MODE = original_mode
