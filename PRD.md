# LC Dual Donchian Rotation Dashboard
## Product Requirements Document · v3.0 · August 2026

**Lighthouse Canton Pte. Ltd. · Internal Research Only · Not for Client Distribution**

---

## 1. Overview

The LC Dual Donchian Rotation Dashboard is an automated, self-updating web dashboard that runs a systematic signal-based rotation strategy across four asset classes: US equities (SPY), Nasdaq-100 (QQQ), Gold (GLD), and Managed Futures (DBMF). The dashboard refreshes daily via GitHub Actions and is served via GitHub Pages.

The strategy uses pairwise Donchian Channel breakouts on price ratios to determine the dominant asset, with a confirmation filter and execution lag to reduce whipsaw. Four execution variants are supported simultaneously.

**Live URL:** https://sgarguk.github.io/donchian-dashboard/  
**Repository:** https://github.com/sgarguk/donchian-dashboard

---

## 2. Strategy Parameters

### Core Parameters

| Parameter | Value | Description |
|---|---|---|
| Universe | SPY, QQQ, GLD, DBMF | Four base assets |
| Execution variants | Base QQQ (1×), QLD (2×), TQQQ (3×), Blend (⅓ each) | Only QQQ leg varies |
| Donchian window (N) | 30 days | Rolling channel lookback |
| Confirmation (CONFIRM) | 2 days | Signal must hold for 2 consecutive trading days |
| Execution shift (SHIFT) | 2 days | Execute 2 trading days after confirmation |
| Transaction cost (TC) | 3bp one-way | Applied at each switch |
| Price data | Raw Close (unadjusted) | Changed from Adj Close Aug 2026 |
| Start date | May 2019 | DBMF inception date |
| Signal type | Majority vote with H2H tiebreaker | 3 pairs vote per asset |

### Pairs and Vote Logic

Six pairwise ratios are computed from the four assets (C(4,2) = 6):

```
SPY/QQQ · SPY/GLD · SPY/DBMF · QQQ/GLD · QQQ/DBMF · GLD/DBMF
```

Each ratio generates a +1 vote for the numerator (if above upper band) or denominator (if below lower band). Maximum 3 votes per asset. The asset with the most votes wins. Ties broken by direct head-to-head ratio.

In-band ratios carry the last confirmed breakout direction — they do not abstain.

---

## 3. Signal Pipeline Logic

### Four States

| State | Condition | Dashboard Display |
|---|---|---|
| Stable | Raw vote winner = current hold | ✓ grey card — no change pending |
| Pending | New challenger, consec < CONFIRM | ⏳ amber card — confirmation progress bar |
| Confirmed | Challenger held CONFIRM days | ⚡ green card — action date shown |
| Executed | SHIFT days elapsed post-confirm | Active Holding card updates |

### Action Date Calculation

Action date = confirmation date + SHIFT trading days, using NYSE calendar (federal holidays + Good Friday).  
Confirmation date = trigger date + (CONFIRM - 1) trading days.

**Example:** trigger 31 Jul (Fri) → confirm 03 Aug (Mon) → action 06 Aug (Wed) at close.

---

## 4. Technical Architecture

### Repository Structure

```
donchian-dashboard/
├── .github/workflows/update.yml     ← GitHub Actions schedule
├── docs/index.html                  ← Auto-generated output (GitHub Pages serves this)
├── dashboard_template.html          ← Static HTML shell with placeholders
├── generate_dashboard.py            ← Main Python script
├── requirements.txt                 ← yfinance, pandas, numpy
└── README.md
```

### Workflow Schedule

| Setting | Value |
|---|---|
| Cron | `0 22 * * 1-5` (10pm UTC = same day as market close, Mon–Fri) |
| Triggers | schedule + workflow_dispatch + push to main |
| Node.js | Forced to Node.js 24 via `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` |
| Push strategy | `git fetch origin` + `git push --force-with-lease` |
| Typical run time | ~25 seconds |

---

## 5. Data Pipeline

### generate_dashboard.py — Processing Steps

1. Download raw close prices via yfinance (`auto_adjust=False`, `Close` column)
2. Compute 6 pairwise ratios and Donchian signals
3. Apply 2-day confirmation filter to reduce whipsaw
4. Build 4 strategy variants (base, qld, tqqq, blend)
5. Compute performance metrics: CAGR, Sharpe, Max DD, Calmar, Sortino
6. Build payload JSON and inject into `dashboard_template.html`
7. Rebuild pairwise table rows from live `curr_ratios`
8. Rebuild vote tally boxes from live `votes`
9. Build signal pipeline card (stable / pending / confirmed)
10. Update header date and price pills
11. Write `docs/index.html` and push to GitHub

---

## 6. Dashboard Sections

| Section | Content | Update Method |
|---|---|---|
| Active Holding | Current asset, days in, since date, previous, switches, prices | JavaScript from payload |
| Signal Pipeline | State, exiting/entering, confirmed date, action date | Python-injected HTML card |
| Pairwise Signal State | All 6 ratios with current/upper/lower/status/vote | Python regex rebuild |
| Vote Tally | SPY/QQQ/GLD/DBMF vote counts with stars | Python regex rebuild |
| Performance | CAGR, Sharpe, Max DD, Calmar vs SPY | JavaScript from payload |
| YTD Equity Chart | Rebased 100, strategy vs SPY vs QQQ, regime background | JavaScript from payload |
| Ratio Charts | 6 pairwise charts with Donchian bands, YTD | JavaScript from payload |
| Annual Returns | Bar chart strategy vs SPY vs QQQ by year | JavaScript from payload |
| YTD Switch Log | All 2025–26 switches with from/to/trigger/days | JavaScript from payload |
| All-Time Switch Log | Full history (no cap) | JavaScript from payload |

---

## 7. Strategy Performance (as of Aug 2026)

### Base QQQ vs SPY — Raw Close, May 2019–Aug 2026

| Metric | Base QQQ | SPY |
|---|---|---|
| CAGR | 27.8% | 16.3% |
| Sharpe | 1.37 | 0.79 |
| Max Drawdown | -18.1% | -33.7% |
| Calmar | 1.53 | 0.48 |
| Total Return | ~490% | ~198% |
| Switches (all-time) | 47 | N/A |

### Variant Comparison

| Variant | QQQ Leg | CAGR | Sharpe | Max DD |
|---|---|---|---|---|
| Base QQQ | 1× | 27.8% | 1.37 | -18.1% |
| QLD | 2× | ~40% | Higher | Deeper |
| TQQQ | 3× | ~52% | Higher | Deepest |
| Blend ⅓ each | ~2× | ~41% | Similar to QLD | Similar to QLD |

---

## 8. Key Design Decisions

### Raw Close vs Adjusted Close
Changed from Adj Close to Raw Close in August 2026. Raw Close matches TradingView and produces better backtest results: +150bps CAGR, better Sharpe, 5 fewer switches over 7 years. SPY dividend adjustments were distorting historical ratios.

### SHIFT=2 vs SHIFT=1
SHIFT=2 retained after backtesting. SHIFT=2 outperforms SHIFT=1 by ~150bps CAGR with marginally better drawdown. The extra day of execution lag reduces whipsaw entries.

### Full Switch Log in Payload
`sw_log` cap removed (was `[-30:]`). Full history passed to payload so the signal pipeline has complete data. The UI display table still shows last 30.

### Python Injection vs Template JS
Pairwise table, vote tally, and signal pipeline are rebuilt in Python and injected via string replace/regex into `new_html`. This keeps the template clean and avoids JavaScript errors from stale hardcoded values.

---

## 9. Known Issues & Future Work

### Resolved
- Pairwise table showing stale hardcoded values → fixed via Python regex rebuild
- Vote tally showing stale values → fixed via Python rebuild
- Action date calculation off by 1–2 days → fixed to calculate from trigger date
- Dashboard not updating (CDN cache) → use incognito or cache-bust URL
- GitHub Actions scheduler unreliable on new repos → resolved after ~1 week

### Pending
- Entry Timing Analysis tab — partially built, deferred to avoid breaking main dashboard
- Cloudflare Pages migration — for private repo hosting
- 7-factor Donchian dashboard — same fixes needed (separate repo)

---

## 10. Operational Notes

### Daily Check
Open https://sgarguk.github.io/donchian-dashboard/ each morning. Header shows "As of [date]". Signal Pipeline card shows current state. If dashboard looks stale, hard refresh (Ctrl+Shift+R) or open in incognito.

### Manual Trigger
GitHub repo → Actions → Update Donchian Dashboard → Run workflow → Run workflow. Useful after market holidays or if scheduled run missed.

### Debugging
If workflow fails: Actions tab → click red X run → expand "Generate dashboard" step → read error. Common causes:
- yfinance DBMF database lock → retry manually
- Python syntax error → check recent edits to generate_dashboard.py
- git push conflict → `git fetch origin` + `git push --force-with-lease`

### Adding Good Friday Dates
The NYSE calendar in `generate_dashboard.py` has Good Friday dates hardcoded. Add future dates annually to the `_good_fridays` list:
```python
_good_fridays = pd.to_datetime([
    '2026-04-03', '2027-03-26', '2028-04-14', ...
])
```

### Storing This Document
Recommended: commit this file to the root of the `donchian-dashboard` repo as `PRD.md` so it is version-controlled alongside the code.

---

*Lighthouse Canton Pte. Ltd. · Internal Research Only · Not for Client Distribution*  
*Document version 3.0 · August 2026 · Prepared by LC Investment Technology*

