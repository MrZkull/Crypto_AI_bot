"""
training_report.py
A single, consistent report format for every train_model.py run — console
output AND a machine-readable JSON alongside it. Import build_report() at
the end of train() in train_model.py and call render_console()/save_json()/
save_markdown() instead of the scattered log.info() calls currently doing
this job piecemeal.

Design goals, based on everything surfaced across prior review rounds:
- Keep the console format you already like (section headers, aligned
  columns) so retrains stay easy to skim in CI logs.
- Add the context that's been MISSING from prior logs and caused real
  confusion: which data window was used, which symbols got full vs.
  partial regime coverage, whether small-n numbers should be trusted,
  and how this run compares to what's currently live.
- Emit the SAME data as JSON so dashboards/gates (like the promotion gate)
  read structured fields instead of re-parsing log text.
"""

import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


SEP = "=" * 60


@dataclass
class SymbolCoverage:
    symbol: str
    regimes_present: list        # e.g. ["recent_bull", "LUNA_crash_May22", ...]
    regimes_missing: list        # regimes with no data for this symbol
    n_test_rows: int
    buy_precision: float
    sell_precision: float

    @property
    def bear_regime_covered(self) -> bool:
        bear_labels = {"LUNA_crash_May22", "FTX_collapse_Nov22", "Bear_trend_Jun22"}
        return bool(bear_labels & set(self.regimes_present))

    @property
    def low_confidence(self) -> bool:
        """Flag symbols where the precision number shouldn't be trusted much yet."""
        return self.n_test_rows < 2000 or not self.bear_regime_covered


@dataclass
class ThresholdRow:
    threshold: float
    n_signals: int
    buy_precision: float
    sell_precision: float
    avg_recall: float
    est_pnl: float


@dataclass
class TrainingReport:
    # Run metadata
    run_note: str = "Manual retrain"
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str = "unknown"
    duration_minutes: float = 0.0

    # Data window disclosure — the sliding-window issue from earlier reviews
    # means every run should say EXACTLY what data it used.
    recent_window_pinned: bool = False
    recent_window_desc: str = "sliding (latest N candles as of run time)"

    # Dataset summary
    n_symbols_requested: int = 0
    n_symbols_with_data: int = 0
    symbols_skipped: list = field(default_factory=list)   # e.g. ["HYPEUSDT"]
    n_rows_total: int = 0
    pct_buy: float = 0.0
    pct_sell: float = 0.0
    pct_no_trade: float = 0.0
    rows_per_regime: dict = field(default_factory=dict)

    # Split info
    embargo_bars: int = 24
    n_train: int = 0
    n_calib: int = 0
    n_test: int = 0
    n_train_after_undersample: int = 0

    # Model performance — held-out test only
    raw_accuracy: float = 0.0
    walk_forward_mean: float = 0.0
    walk_forward_std: float = 0.0
    buy_precision: float = 0.0
    buy_recall: float = 0.0
    sell_precision: float = 0.0
    sell_recall: float = 0.0

    # Threshold sweep (tuned on calibration split, reported on untouched test —
    # see per-symbol/threshold-leak fix from earlier review rounds)
    threshold_sweep: list = field(default_factory=list)     # list[ThresholdRow]
    selected_buy_threshold: float = 0.45
    selected_sell_threshold: float = 0.45

    # Per-symbol breakdown + coverage flags
    per_symbol: list = field(default_factory=list)          # list[SymbolCoverage]

    # Comparison vs currently-live model (promotion-gate context)
    live_buy_precision: Optional[float] = None
    live_sell_precision: Optional[float] = None

    # Feature engineering sanity
    engineered_features_selected: list = field(default_factory=list)
    engineered_features_dropped: list = field(default_factory=list)

    # ── Derived / computed properties ──────────────────────────────────

    @property
    def low_confidence_symbols(self) -> list:
        return [s.symbol for s in self.per_symbol if s.low_confidence]

    @property
    def buy_precision_delta(self) -> Optional[float]:
        if self.live_buy_precision is None:
            return None
        return round(self.buy_precision - self.live_buy_precision, 4)

    @property
    def sell_precision_delta(self) -> Optional[float]:
        if self.live_sell_precision is None:
            return None
        return round(self.sell_precision - self.live_sell_precision, 4)

    # ── Output renderers ─────────────────────────────────────────────────

    def render_console(self) -> str:
        lines = []
        a = lines.append

        a(SEP)
        a(f"TRAINING REPORT — {self.trained_at}")
        a(f"Note: {self.run_note}  |  Commit: {self.git_commit[:8]}  |  Duration: {self.duration_minutes:.1f} min")
        a(SEP)

        a("\nDATA WINDOW")
        if self.recent_window_pinned:
            a(f"  PINNED — {self.recent_window_desc}  (safe for run-to-run comparison)")
        else:
            a(f"  SLIDING — {self.recent_window_desc}")
            a("  ⚠ Not pinned: this run's 'recent' data differs from every other run's.")
            a("    Don't compare precision numbers across runs as if they used the same test set.")

        a("\nDATASET")
        a(f"  Symbols requested: {self.n_symbols_requested}  |  With data: {self.n_symbols_with_data}")
        if self.symbols_skipped:
            a(f"  ⚠ SKIPPED (no data — check fetch source per-symbol): {', '.join(self.symbols_skipped)}")
        a(f"  Total rows: {self.n_rows_total:,}")
        a(f"    BUY:      {self.pct_buy*100:5.1f}%")
        a(f"    SELL:     {self.pct_sell*100:5.1f}%")
        a(f"    NO_TRADE: {self.pct_no_trade*100:5.1f}%")
        a("  Rows per regime:")
        for regime, cnt in self.rows_per_regime.items():
            a(f"    {regime:<28} {cnt:>9,}")

        a("\nSPLIT (embargoed per symbol+regime)")
        a(f"  embargo={self.embargo_bars} bars  train={self.n_train:,}  calib={self.n_calib:,}  test={self.n_test:,}")
        a(f"  train after undersample: {self.n_train_after_undersample:,}")

        a("\nHELD-OUT TEST PERFORMANCE  (untouched by training AND threshold tuning)")
        a(f"  Raw accuracy (misleading — dominated by NO_TRADE): {self.raw_accuracy*100:.1f}%")
        a(f"  Walk-forward: {self.walk_forward_mean*100:.1f}% ± {self.walk_forward_std*100:.1f}%")
        a(f"  BUY:  precision={self.buy_precision*100:5.1f}%  recall={self.buy_recall*100:5.1f}%")
        a(f"  SELL: precision={self.sell_precision*100:5.1f}%  recall={self.sell_recall*100:5.1f}%")

        if self.live_buy_precision is not None:
            a("\nVS. CURRENTLY-LIVE MODEL")
            bd, sd = self.buy_precision_delta, self.sell_precision_delta
            a(f"  BUY  precision: {self.buy_precision*100:.1f}% vs live {self.live_buy_precision*100:.1f}%  ({bd*100:+.1f} pts)")
            a(f"  SELL precision: {self.sell_precision*100:.1f}% vs live {self.live_sell_precision*100:.1f}%  ({sd*100:+.1f} pts)")

        if self.threshold_sweep:
            a("\nTHRESHOLD SWEEP  (tuned on calibration split — see selected values below)")
            a(f"  {'Thresh':<8}{'n':>8}{'BUY Prec':>10}{'SELL Prec':>11}{'Recall':>9}{'Est PnL':>12}")
            for r in self.threshold_sweep:
                a(f"  {r.threshold:<8.2f}{r.n_signals:>8}{r.buy_precision*100:>9.1f}%{r.sell_precision*100:>10.1f}%"
                  f"{r.avg_recall*100:>8.1f}%{r.est_pnl:>12,.0f}")
            a(f"  → Selected BUY threshold:  {self.selected_buy_threshold:.2f}")
            a(f"  → Selected SELL threshold: {self.selected_sell_threshold:.2f}")

        if self.per_symbol:
            a("\nPER-SYMBOL BREAKDOWN")
            a(f"  {'Symbol':<14}{'n':>7}{'BUY Prec':>10}{'SELL Prec':>11}  Flags")
            for s in sorted(self.per_symbol, key=lambda x: x.symbol):
                flags = []
                if s.low_confidence:
                    flags.append("LOW-N/NO-BEAR-DATA")
                flag_str = ", ".join(flags)
                a(f"  {s.symbol:<14}{s.n_test_rows:>7}{s.buy_precision*100:>9.1f}%{s.sell_precision*100:>10.1f}%  {flag_str}")

            if self.low_confidence_symbols:
                a(f"\n  ⚠ Treat these symbols' numbers as unreliable until more data accumulates:")
                a(f"    {', '.join(self.low_confidence_symbols)}")

        if self.engineered_features_selected or self.engineered_features_dropped:
            a("\nENGINEERED FEATURES")
            a(f"  Selected: {', '.join(self.engineered_features_selected) or '(none)'}")
            a(f"  Dropped:  {', '.join(self.engineered_features_dropped) or '(none)'}")

        a("\n" + SEP)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["low_confidence_symbols"] = self.low_confidence_symbols
        d["buy_precision_delta"] = self.buy_precision_delta
        d["sell_precision_delta"] = self.sell_precision_delta
        return d

    def save_json(self, path: str = "model_performance.json") -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_markdown(self, path: str = "MODEL_CARD.md") -> None:
        """Human-readable run history — append each run as a new section so
        you can scroll back through past retrains in one file."""
        md = []
        md.append(f"## Run: {self.trained_at}\n")
        md.append(f"- **Note:** {self.run_note}")
        md.append(f"- **Commit:** `{self.git_commit[:8]}`")
        md.append(f"- **Duration:** {self.duration_minutes:.1f} min")
        md.append(f"- **Data window:** {'PINNED' if self.recent_window_pinned else 'sliding'} — {self.recent_window_desc}")
        md.append(f"- **Rows:** {self.n_rows_total:,} (BUY {self.pct_buy*100:.1f}% / SELL {self.pct_sell*100:.1f}% / NO_TRADE {self.pct_no_trade*100:.1f}%)")
        md.append(f"- **BUY:** precision {self.buy_precision*100:.1f}%, recall {self.buy_recall*100:.1f}%")
        md.append(f"- **SELL:** precision {self.sell_precision*100:.1f}%, recall {self.sell_recall*100:.1f}%")
        if self.live_buy_precision is not None:
            md.append(f"- **Vs. live:** BUY {self.buy_precision_delta*100:+.1f} pts, SELL {self.sell_precision_delta*100:+.1f} pts")
        if self.symbols_skipped:
            md.append(f"- **Skipped symbols:** {', '.join(self.symbols_skipped)}")
        if self.low_confidence_symbols:
            md.append(f"- **Low-confidence symbols:** {', '.join(self.low_confidence_symbols)}")
        md.append("")

        # Prepend (newest run first) if file already exists
        existing = ""
        try:
            with open(path) as f:
                existing = f.read()
        except FileNotFoundError:
            existing = "# Model Training History\n\n"

        header, _, rest = existing.partition("\n\n")
        with open(path, "w") as f:
            f.write(header + "\n\n" + "\n".join(md) + "\n" + rest)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    # Worked example reproducing the shape of your Jul-14 log, to show the
    # template's output format end-to-end.
    report = TrainingReport(
        run_note="Manual retrain",
        git_commit=get_git_commit(),
        duration_minutes=22.9,
        recent_window_pinned=False,
        n_symbols_requested=26,
        n_symbols_with_data=25,
        symbols_skipped=["HYPEUSDT"],
        n_rows_total=417438,
        pct_buy=0.219, pct_sell=0.247, pct_no_trade=0.533,
        rows_per_regime={
            "recent_bull": 124400, "Recovery_Jan23": 59976, "Bull_peak_Oct21": 57120,
            "Bear_trend_Jun22": 57120, "Aug2023_dip": 30383, "Apr2024_halving": 30383,
            "FTX_collapse_Nov22": 29736, "LUNA_crash_May22": 28320,
        },
        embargo_bars=24, n_train=270954, n_calib=62614, n_test=83486,
        n_train_after_undersample=257486,
        raw_accuracy=0.540, walk_forward_mean=0.448, walk_forward_std=0.042,
        buy_precision=0.489, buy_recall=0.107, sell_precision=0.465, sell_recall=0.117,
        selected_buy_threshold=0.45, selected_sell_threshold=0.45,
        threshold_sweep=[
            ThresholdRow(0.35, 9981, 0.466, 0.452, 0.118, 376498),
            ThresholdRow(0.40, 9706, 0.473, 0.459, 0.117, 385855),
            ThresholdRow(0.45, 8634, 0.489, 0.465, 0.107, 372730),
            ThresholdRow(0.50, 5813, 0.552, 0.475, 0.077, 314490),
        ],
        per_symbol=[
            SymbolCoverage("APTUSDT", ["recent_bull", "FTX_collapse_Nov22", "Aug2023_dip", "Apr2024_halving", "Recovery_Jan23"],
                            ["LUNA_crash_May22", "Bear_trend_Jun22", "Bull_peak_Oct21"], 2377, 0.283, 0.312),
            SymbolCoverage("BTCUSDT", ["recent_bull", "LUNA_crash_May22", "FTX_collapse_Nov22", "Bear_trend_Jun22",
                                        "Aug2023_dip", "Apr2024_halving", "Bull_peak_Oct21", "Recovery_Jan23"],
                            [], 3802, 0.555, 0.573),
        ],
        engineered_features_selected=["hour_sin", "hour_cos", "dow_sin", "dow_cos", "btc_corr_20", "btc_beta_20"],
        engineered_features_dropped=["taker_buy_ratio", "btc_rel_strength"],
        live_buy_precision=0.512, live_sell_precision=0.498,
    )

    print(report.render_console())
    report.save_json("model_performance.json")
    report.save_markdown("MODEL_CARD.md")
    print("\n(Also wrote model_performance.json and MODEL_CARD.md)")
