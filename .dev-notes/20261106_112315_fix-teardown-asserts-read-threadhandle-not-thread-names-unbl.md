---
id: 20261106_112315_fix-teardown-asserts-read-threadhandle-not-thread-names-unbl
status: done
board: 20261106_112315_fix-teardown-asserts-read-threadhandle-not-thread-names-unbl
title: "fix: teardown asserts read ThreadHandle not thread names — unblocks test-core-py red on main (PR #294)"
tags: ["testing", "infra", "ci"]
sprint: ""
gh: null
---

# fix: teardown asserts read ThreadHandle not thread names — unblocks test-core-py red on main (PR #294)

f2f91016a stage-3B moved idle loop to spawn_loop(); name-scan asserts went vacuous (pass even while loop runs) + 2 fail on pristine main. _idle_alive(mgr) = ThreadHandle.is_alive() OR thread scan; 3 sites; negative control proven (leak->True, stopped->False); 8 passed + 71 closure; test-only, no prod change. Unblocks test-core-py for #265/#266.
