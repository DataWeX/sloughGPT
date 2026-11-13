---
title: "Native tokenizer: unify char + token-tree into ONE algorithm (policy hoisted)"
date: 2026-10-13
tags: [tokenizer, native, char, token-tree, consolidation, refactor]
status: done
---

# Native tokenizer: unify char + token-tree into ONE algorithm

## Ask (user, 2026-10-13)

> "we end the chat and token to be on each other, characters makes up to tokens.
> so we dont need separate char and token in native tokenizer implementation."

Correct. Verified in code, not assumed.

## Finding — the trie-walk core is IDENTICAL

`TokenTree.query()` (domain/training/_internal/token_tree.py:298) does greedy
longest-prefix match over the encoding trie. Over a **depth-1 trie** (char-only
vocab, zero merges) that reduces to exactly `stoi.get(c, 0)` — the legacy
`_CharTokenizer.encode` body. One algorithm, two configurations. `char` is NOT a
peer of `tree`; it is a **degenerate tree**.

`NativeTokenizer.encode` (slonet_provider.py) currently branches on
`self.variant`, so the two implementations have drifted into two code paths over
one algorithm. That is the duplication to remove.

## But it is FIVE policy knobs, not one

Measured divergence between the pinned legacy behaviors (see gate below):

| policy        | char (legacy, PINNED) | tree                |
|---------------|-----------------------|---------------------|
| pretokenizer  | none — raw chars      | gpt2 regex          |
| lowercase     | False                 | True                |
| word_suffix   | `""`                  | `</w>`              |
| **OOV ->**    | **id 0**              | `unk_id` (1)        |
| decode        | raw join              | skips specials, emits spaces at `</w>` |

The OOV row is the trap: a naive degenerate tree returns `unk_id` on unknown
input; legacy char returns **0**. `test_oov_char_maps_to_zero` pins
`encode("z") == [0]`. Unify without hoisting `oov_id` as policy and every
existing char-vocab model silently changes ids — the same silent-corruption
class as the `SessionKVCache` bug, and the same one the tokenizer battery
surfaced (`'1234567890 98765' -> '12345 5'`, no error raised).

Decode is NOT the trie walk — its complexity exists because of `word_suffix`.
With `word_suffix=""` most of it falls away; special-token skipping needs its
own policy flag.

## Landmine to fix while unifying

`query()` does `if ch == "<": ... break` for multi-piece specials and **never
falls through** to single-char lookup. A real tree is unaffected (`<` alone is
not an edge). A char vocab CONTAINING `"<"` would start returning OOV. Safe
generalization: when no multi-piece matches, fall through to
`node.children.get(ch)`.

## Plan

1. `TokenTree` gains a policy object: `pretokenizer=None`, `lowercase=False`,
   `word_suffix=""`, `oov_id=0`, `skip_special_on_decode=False`.
2. `TokenTree.query` uses `self._oov_id` instead of hard `unk_id`; falls through
   on unmatched `<`.
3. `TokenTree.encode` honours policy (no lowercase, no suffix, no pretokenize).
4. `TokenTree.decode` honours policy (raw join when `word_suffix=""` and
   `skip_special=False`).
5. `NativeTokenizer` DELETES the `self.variant` branch in encode/decode and
   delegates to the tree. `variant` stays as a provenance LABEL only
   (`test_one_scheme_per_family_and_uniform_attrs` asserts `native.variant ==
   "char"` and `native.pretokenizer is None`).
6. `_char_carrier_tree` builds a REAL depth-1 trie via `_add_token` instead of an
   empty carrier.
7. `to_dict`/`from_dict` round-trip the new policy fields with defaults, so
   legacy `.soul` files (no policy keys) still load byte-identically.

## GATE — existing tests are the acceptance criterion

`tests/core-py/test_native_tokenizer_format.py` (278 lines) already asserts char
AND tree stay **byte-identical to the legacy wrappers**, against INDEPENDENT
inline oracles (`_legacy_char_encode`, `_legacy_char_decode` frozen in the test
body — not the class names, which are aliases and would be vacuous):

- `test_encode_byte_identical` / `test_decode_roundtrip_byte_identical`
- `test_decode_arbitrary_ids_byte_identical` (gaps + unknown ids -> "")
- `test_oov_char_maps_to_zero`  <- the trap
- `test_case_is_preserved`
- `test_lowercase_normalization_active` (tree side)
- `test_save_stamp_prefers_tree_then_char`, `test_format_literals_agree...`

If the unification is behaviour-preserving these pass. If it is not, they fail
loudly. Do NOT weaken them.

## Also in scope (follow-on, separate commit)

`SloUnigram._all_segmentations` enumerates EVERY segmentation into a list —
exponential in pretoken length. One 48-char unbroken word
(`lamavirtualassistantdesignedtoprovideinformation`) hangs/OOMs `train()`. Real
corpus with any URL / base64 / identifier hits this. Measured: 0.10s on short
pretokens vs minutes on one long one.

## Benchmark

Re-run /tmp/opencode/tok_bench.py as THREE rows (not five) — char folds into
tree as a config. Prior numbers for comparison:

| candidate             | F1 full | F2 stream | F5 rt/s   |
|-----------------------|---------|-----------|-----------|
| MorphTokenizer        | 11/11   | 11/11     | 1,619     |
| SloBPE                | 3/11    | 3/11      | 157,002   |
| SloUnigram            | 4/11    | 4/11      | 214,469   |
| NativeTokenizer(char) | 4/11    | 4/11      | 3,633,492 |
| NativeTokenizer(tree) | 3/11    | 3/11      | 609,674   |

F1/F2 shortfalls were OOV coverage (tiny corpus), NOT decode bugs — verified.
So the rebenchmark must judge the refactor on (a) parity tests green and
(b) char-config throughput vs the old 3.6M rt/s, not on raw F1.

## SHARED WORKSPACE

index updated - required if any doc lands (docs/INDEX.md)
shared copy reused - yes, one TokenTree, no new module, no new dep
coordination named - tests/core-py/test_native_tokenizer_format.py is the gate;
                     domain/inference/_internal/slonet_provider.py + domain/
                     training/_internal/token_tree.py both CLEAN as of 2026-10-13
