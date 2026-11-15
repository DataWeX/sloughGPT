---
id: 20261106_104845_avion-never-offer-a-card-the-finder-cannot-resolve
status: todo
board: 20261106_104845_avion-never-offer-a-card-the-finder-cannot-resolve
title: "avion: never offer a card the finder cannot resolve"
tags: ["avion", "choice", "area:agents"]
sprint: ""
gh: null
---

# avion: never offer a card the finder cannot resolve

propose_options grounds candidates from the accessibility tree, but execution resolves the locator through ElementFinder, and nothing checks that the two agree. Observed on a fake: 'More' is advertised in the tree, offered as a card, and then errors 'Element not found' at click time.

Fix in production, not the fixture: filter proposed candidates to locators the finder can actually resolve, so the model is never handed an action guaranteed to fail.

Scope warning: this changes which cards the model ever sees, so it must land as its own commit behind the suite, not folded into a parsing change. The test fixture (TreeBackend advertising More without a backing element) is being fixed separately to unblock the branch.
