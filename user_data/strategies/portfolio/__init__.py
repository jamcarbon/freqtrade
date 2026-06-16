"""Portfolio layer - runs strategy #2 (momentum) and #3 (funding carry) in parallel.

Two INDEPENDENT $10k books (combined $20k), each its own validated one-engine-two-
feeds stack. This layer adds:
  * combined.py  - parallel replay + the diversification report (does #2+#3 beat
                   either alone?) and a CombinedRiskMonitor (net-beta + total-DD).
  * run_paper.py - orchestrates both books' forward-test ticks + a combined log.

Nothing here launches live trading; both books' live brokers remain gated.
"""
