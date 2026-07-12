"""
.slnc — SloughGPT Neural Cache format.

A memory-mapped inference format with computation-graph-aligned layout.
Designed as a compiler pipeline: spec → compile → link → load.

Architecture:
  spec.py    — binary format specification (schema)
  compiler.py — safetensors → .slnc compilation
  parser.py  — .slnc → memory-mapped loading

Format overview:
  [magic: "SLNC"]
  [header: fixed-size metadata]
  [tensor_table: offset/size entries for all tensors]
  [tensor_data: weights in computation order]
  [checksums: CRC32 per tensor for integrity]

Key properties:
  - Zero-copy: numpy arrays are views into mmap'd pages
  - Computation order: Block 0 → Block 1 → ... → norm → lm_head
  - Demand loading: OS pages in only accessed blocks
  - Integrity: per-tensor CRC32 checksums
  - Extensible: version field for format evolution
"""
