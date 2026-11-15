---
id: 20261115_092427_land-dep-bump-batch-pr-319-332-14-dep-bumps-337-grib-slug-fi
status: wip
board: 20261115_092427_land-dep-bump-batch-pr-319-332-14-dep-bumps-337-grib-slug-fi
title: "Land dep-bump batch: PR #319-332 (14 dep bumps) + #337 (grib slug fix) green-only"
tags: ["ci", "deps"]
sprint: ""
gh: "319-337"
---

# Land dep-bump batch: PR #319-332 (14 dep bumps) + #337 (grib slug fix) green-only

Approved batch. Protocol per PR: update-branch onto current main (all 14 have stale 11-13 reds: test-core-py + Board integrity from the ubuntu-26.04 runner era, not reproducible on pristine green main), fresh CI green, fresh mergeable re-check, merge. Never merge red. #337 (grib card-id slug fix) CI fresh already. Skip+report any PR still red after update or newly-conflicting sibling (lockfile bumps).
