"""Retired legacy emergency-close script.

This script previously authenticated directly with requests, submitted market
orders, cancelled all orders by currency, and cleared local trade records even
when exchange operations failed. That bypasses the reviewed Deribit client,
order-intent ledger, environment guard and verified close workflow.

It is intentionally disabled. Do not restore direct exchange calls here. A
future emergency control must be implemented through the approved execution
coordinator, with a separately authorised Testnet-only kill-switch procedure.
"""


def main() -> None:
    raise SystemExit(
        "emergency_close.py is retired and cannot access the exchange. "
        "Do not clear trades.json manually. Reconcile the exchange positions, "
        "open orders and local ledger through the reviewed operator procedure."
    )


if __name__ == "__main__":
    main()
