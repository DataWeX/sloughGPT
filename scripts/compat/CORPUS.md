# Compat corpus — third-party programs run against the x86 VM

Vendored sources for `scripts/compat_track.py`. All are third-party,
permissively licensed, and unmodified (intake happens exactly as-upstream —
if our assembler rejects a construct, that is the finding, not a defect to
patch around).

| entry              | origin                                                       | commit                                   | license | license file                    |
| ------------------ | ------------------------------------------------------------ | ---------------------------------------- | ------- | ------------------------------- |
| bootmine.asm       | https://github.com/io12/BootMine (`mine.asm`)                | d784110b837e1b4d9b2b955f745ebe4269b55e88 | MIT     | LICENSE.io12_BootMine           |
| bweeper.asm        | https://github.com/spiritinchains/bweeper (`bweeper.asm`)    | fe25fa938de94bb657a1dc597eaad389d767371d | MIT     | LICENSE.spiritinchains_bweeper  |
| nasm_tetris.asm    | https://github.com/Arkowski24/nasm-tetris (`src/tetris.asm`) | 3fc95987c68e71200538b2f52f1b032e9b15b85f | MIT     | LICENSE.Arkowski24_nasm-tetris  |
| asteroids_main.asm | https://github.com/adaxiik/asteroids-asm (`src/main.asm`)    | 6e1ab900c0d378367ea66c3b8ed1eccb818efb65 | MIT     | LICENSE.adaxiik_asteroids-asm   |
| snake_main.asm     | https://github.com/NikitaIvanovV/snake-asm (`snake.asm`)     | b695abd0d973fff4cffde9cf83fcb5c5968029fb | MIT     | LICENSE.NikitaIvanovV_snake-asm |

Notes:

- Multi-file projects are vendored as their **main file only**; upstream
  sibling files (`externs.asm`, `utils.asm`, `*.mac`) are deliberately not
  stitched in — our intake has no include/macro path, and pretending
  otherwise would hide the gap the matrix exists to measure.
- MIT license texts are vendored verbatim as `LICENSE.<repo>` per the
  redistribution terms.
- Corpus membership is agent-chosen (permissive + assembleable-by-some
  NASM-like toolchain); replacing or extending entries is expected as the
  intake path improves.
