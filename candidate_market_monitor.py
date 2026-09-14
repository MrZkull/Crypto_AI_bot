#candidate_market_monitor.py

import asyncio, json, time
import websockets
from filelock import FileLock
from deribit_client import DeribitClient
from execution_policy import evaluate_state_transition, execute_tp1_partial, calculate_continuous_funding, evaluate_thesis_outcome

def log_event(event_type: str, candidate_id: str, pred_id: str, data: dict):
    with FileLock("candidate_events.jsonl.lock"):
        with open("candidate_events.jsonl", "a") as f:
            f.write(json.dumps({"local_ts": int(time.time()*1000), "event": event_type, 
                                "candidate_id": candidate_id, "pred_id": pred_id, **data}) + "\n")

async def shadow_stream():
    deribit = DeribitClient(...)
    
    async with websockets.connect("wss://test.deribit.com/ws/api/v2") as ws:
        # Initial subscription payload...
        
        async for msg in ws:
            data = json.loads(msg)
            if "params" not in data: continue
            
            ticker = data["params"]["data"]
            sym = ticker["instrument_name"].split("_")[0] + "USDT"
            exchange_ts = int(ticker["timestamp"])
            
            best_bid, best_ask = float(ticker["best_bid_price"]), float(ticker["best_ask_price"])
            mark_price = float(ticker["mark_price"])
            current_funding = float(ticker.get("current_funding", 0.0))
            
            with FileLock("candidate_state.json.lock"):
                try:
                    state = json.load(open("candidate_state.json"))
                except FileNotFoundError:
                    continue
                    
                state_changed = False
                
                for pid, t in list(state.items()):
                    if t["symbol"] != sym or t["status"] not in ("ACTIVE", "COUNTERFACTUAL_ACTIVE"): continue

                    last_ts = int(t.get("last_funding_ts", exchange_ts))
                    elapsed_ms = exchange_ts - last_ts
                    
                    if elapsed_ms < 0 or elapsed_ms > 5000:
                        t["exit_reason"] = "DATA_INVALIDATION"
                        t["status"] = "CLOSED"
                        funding_usd = float(t.get("funding_usd", 0.0))
                        
                        log_event("OUTCOME_OBSERVED", t["candidate_id"], pid, {
                            "reason": "DATA_INVALIDATION",
                            "gap_ms": int(elapsed_ms),
                            "realized_gross_pnl": 0.0,
                            "final_exit_fee_usd": 0.0,
                            "funding_usd": funding_usd
                        })
                        state_changed = True
                        continue

                    t["funding_usd"] = t.get("funding_usd", 0.0) + calculate_continuous_funding(
                        t["side"], t["qty"], mark_price, current_funding, elapsed_ms
                    )
                    t["last_funding_ts"] = exchange_ts

                    is_buy = t["side"] == "BUY"
                    exec_exit = best_bid if is_buy else best_ask

                    new_state, action = evaluate_state_transition(
                        t["side"], t["state"], exec_exit, t["simulated_entry"], t["stop"], t["tp1"], t["tp2"]
                    )
                    
                    if action == "HIT_TP1":
                        try: exit_fee_rate = deribit.get_effective_fee_rate(sym, "taker")
                        except Exception: continue
                            
                        t = execute_tp1_partial(t, exec_exit, exit_fee_rate, partial_pct=0.5)
                        t["state"] = "PARTIAL_TP1"
                        t["stop"] = t["simulated_entry"]
                        log_event("PARTIAL_TP1_FILLED", t["candidate_id"], pid, 
                                  {"fill_price": exec_exit, "realized_gross": t["realized_gross_pnl"], "tp1_fee": t["tp1_fee_usd"]})
                        state_changed = True

                    elif action in ("HIT_SL", "HIT_TP2", "AMBIGUOUS_GAP_ASSUME_SL"):
                        try: exit_fee_rate = deribit.get_effective_fee_rate(sym, "taker")
                        except Exception: continue
                            
                        t["status"] = "CLOSED"
                        t["exit_price"] = exec_exit
                        t["final_exit_fee_usd"] = (t["qty"] * exec_exit) * exit_fee_rate
                        
                        gross_pnl = (exec_exit - t["simulated_entry"]) * t["qty"]
                        if not is_buy: gross_pnl *= -1
                        t["realized_gross_pnl"] = t.get("realized_gross_pnl", 0.0) + gross_pnl
                        t["exit_reason"] = action
                        
                        log_event("OUTCOME_OBSERVED", t["candidate_id"], pid, {
                            "realized_gross_pnl": t["realized_gross_pnl"],
                            "entry_fee_usd": t.get("entry_fee_usd", 0.0),
                            "tp1_fee_usd": t.get("tp1_fee_usd", 0.0),
                            "final_exit_fee_usd": t["final_exit_fee_usd"],
                            "funding_usd": t["funding_usd"],
                            "reason": action
                        })
                        state_changed = True
                    
                    elif new_state != t["state"]:
                        if new_state == "TRAILING_TP1": t["stop"] = t["tp1"]
                        t["state"] = new_state
                        log_event("STATE_CHANGE", t["candidate_id"], pid, {"new_state": new_state})
                        state_changed = True

                if state_changed:
                    with open("candidate_state.json.tmp", "w") as f: json.dump(state, f, indent=2)
                    import os; os.replace("candidate_state.json.tmp", "candidate_state.json")

if __name__ == "__main__":
    asyncio.run(shadow_stream())
