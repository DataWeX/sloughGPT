---
title: Full Recovered Conversation — Boomerang Hash Tree
date: 2026-08-31
tags: [planner, kanban, hashtree, boomerang, recovered, conversation]
status: done
---

# Full Recovered Conversation — Boomerang Hash Tree

## Session: Oon Kanban board demo (ses_faa3fe4b9ffeR4MxXIHB2nXth8)

### Message 1 (User): Building the hashing system
"lets build some software engineering for when we vectorize our note tasking system... hash changes when note changes... how about we building our hash system, to start from the cards hash... but first let's think of our kanban as food, for babies, then the cards would be food stored inside Docks called their Slots on Trays called pards"

### Message 2 (Assistant): Proposed architecture
Tray = Board/Column, Slot = Position in column, Card = Task/note, Card hash = Content hash

### Message 3 (User): Clarifying the metaphor
"Slots can be shuffled, vectorize everything, then hash it out build our algorithm to be a boomeranging algorithm... boards is called tray, the bakery is the table, and the slot is called..."

### Message 4 (Assistant): Updated architecture
Board → Tray, Table → Data store, Slot = Position (shufflable)

### Message 5 (User): The buffet metaphor
"we have the gui is the buffets for eating foods, so many people can have their own buffets to serve food... we have our section called our table buffet, lets call that our account on the big buffet queue, the table is our task board, there are trays on it called states called picks/choices/mades/themes"

### Message 6 (Assistant): Full metaphor map
Buffet = GUI/App, Table = Account/User board, Tray = State column (picks/choices/mades/themes), Slot = Position in tray, Card = Task/note item

### Message 7 (User): Shuffling and state tracking
"yes so trays can be shuffled, they have slots that can now be stuck in there theme or state or else they rot away and shed skin... when a state has been moved we keep track of it but turn around and vanishing it go happen"

### Message 8 (User): Boomerang algorithm explained
"hashes can be generated from colors so think of our algorithm as boomeranging between colors and then having a swatch of its own... cards on multiple trays have them, since they have a mother that holds the cards and slots... shuffling is just when we are allowed to talk about the state of each tray without owning it in the hashes, so hashes dont change and we cant shuffle outside the tray because it has a owner"

### Message 9 (User): Semantic preservation
"break it down sentence by sentence and recreate it, and wonder how you did that without breaking my words into another meaning, see? it's all hashes... just think of our hashing boomeranging algorithm as tied into vector graphing states of storing the planner manipulations and transformations of their data into code without losing their quality or their semantic accuracy"

### Message 10 (User): Color swatches
"we would have color swatches we would create hashes from onto pixels, we will discuss the vector and algorithm later get the structure first"

### Message 11 (Assistant): Boomerang = encryption layer
```
Data → Hash (encrypted, can't read) → Vector space (just a line, no data)
                                              ↓
                                    Search by similarity (hash-level)
                                              ↓
                                    Boomerang back → verify hash → show data
```

## Key Concepts

| Concept | Meaning |
|---------|---------|
| **Boomerang** | Hash → Vector → Similarity → Back to hash (encryption/verification) |
| **Mother hash** | Root hash that holds cards and slots (never changes) |
| **Design** | Semantic fingerprint of the card |
| **Swatch** | Color that generates a hash → pixel |
| **Shuffling** | Moving cards between trays without breaking hash chain |
| **Sentence decomposition** | Break down, transform, reconstruct without losing meaning |
| **Typing errors** | System preserves semantic accuracy through typos |

## Architecture

```
Mother Hash (root, encrypted)
├── Card Hash (semantic identity, encrypted)
├── Note Hash 1
├── Note Hash 2
└── Slot Ref (tray + position + time)

Tray = state, owns slots (shuffle OK, hash unchanged)
Card = semantic entity (move between trays, hash unchanged)
Hash = content changes → hash changes
Boomerang = hash verifies → decrypt → show data
```

## What was built
- `packages/planner/src/planner/hashtree.py` — Full implementation with:
  - CardSlotHash (root, card_id, slot_id, tray, position, color, design)
  - NoteHash (note_id, hash_value, root_ref, version)
  - HashHistoryEntry (audit trail)
  - HashCommit (immutable commits like git, with color/pixel)
  - HashTree (compute, derive, on_note_add/edit/delete, get_all_hashes, verify, vectorize)
  - HashTreeStore (persistence)

## Color swatches (from ThemeSwatch component)
- Periwinkle: #8b7bc4
- Lilac: #a67fd4
- Rose: #d894b4
- Coral: #e88890
- Peach: #e8a86c
- Mint: #6bb89a
- Dew: #6cabcc

## Baby pastel theme (from CardEditor.tsx)
- rose: #f4b8c8
- lavender: #c8b8f4
- sky: #b8d8f4
- mint: #b8f4d8
- peach: #f4d8b8
- lilac: #d8b8f4
- cream: #f4f0e8
- blush: #f8e8f0
- fog: #e8e4f0
- ink: #3a3450
- muted: #8a82a0
