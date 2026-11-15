---
id: 20261109_112817_grib-survey-leftovers-manifest-card-count-counts-the-schema
status: done
board: 20261109_112817_grib-survey-leftovers-manifest-card-count-counts-the-schema
title: "grib survey leftovers: manifest card_count counts the schema header line; suite coverage 71% (kanban.py 39%)"
tags: ["grib", "tests"]
sprint: ""
gh: null
---

# grib survey leftovers: manifest card_count counts the schema header line; suite coverage 71% (kanban.py 39%)

Survey 2026-11-09 (grib quality pass, leftovers F5+F6). F6: export() card_count = count of ALL non-empty lines including the planner/1 schema header — 174 reported vs 173 real cards (verified 2026-11-09 against the live store); informational only, the freshness gate compares board_sha256 so unaffected; fix = count dicts carrying an id (watch: scripts/check_kanban_freshness.py may display it). F5: suite coverage 71% (gate is 50%); weakest modules kanban.py 39% (525 missed statements — the F1 chain bug lives exactly in this blind zone), cli.py 63%, core.py 67%; raise coverage with behavior tests only where it matters (no coverage-chasing; F1 tests will lift kanban.py on their own).
