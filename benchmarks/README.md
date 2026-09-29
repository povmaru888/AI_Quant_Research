# Panel performance and PIT market-value evidence

`panel_phase_a_full_profile.json` records the optimized 72-month v2 rebuild;
`panel_golden_comparison.json` records its strict comparison with the existing
panels; and `panel_cache_hit.json` records the no-op cache run. The final
profile is from the same local SQLite database after the market-value ETL.

The legacy timing sample uses the unchanged `tools/build_panels.py` from HEAD
on six representative months. Its 72-month legacy wall time is estimated in
`panel_performance_and_pit_gate.json`: the six measured month times are scaled
to 72 months and the one shared full-price-load overhead is added once. The
full legacy 72-month build was not run. Peak working sets were measured in each
builder process.

`pit_v3_market_value_coverage.json` contains the full PIT coverage check and
the missing stock-date details. Coverage reaches 99.012%, but 92 estimated
40–60 億 TWD near-threshold stock-date cases have no direct PIT value. The plan's
100% near-threshold requirement therefore blocks v3 panel/model/OOS publication.
