---
id: 20261114_151505_dataset-adapter-registry-replace-format-if-chain-with-sniffe
status: todo
board: 20261114_151505_dataset-adapter-registry-replace-format-if-chain-with-sniffe
title: "Dataset adapter registry: replace format if-chain with sniffers dispatching to adapters"
tags: ["training", "architecture"]
sprint: ""
gh: null
---

# Dataset adapter registry: replace format if-chain with sniffers dispatching to adapters

detect_dataset_type (training.py) is a growing if-chain (messages/conversation/instruction/audio/image/video/code/...) and sampler wiring is hand-matched per source (ExperienceSampler, FeedBatchSampler, chat/instruction samplers, train/val split). General-purpose trainer needs declared data sources: (1) registry of sniffers — each format sniff returns an adapter (not just a label); (2) each adapter declares its BatchSampler contract (pairs or token stream + block_size); (3) registration = one registry entry (same microkernel pattern as router manifest); (4) image/video sniffers must declare capabilities.unsupported rather than fake-success — IMAGE_TEXT detection today labels data nothing can consume. Existing formats grandfathered: the if-chain shrinks as sniffers migrate, never a big-bang rewrite. Gate: format-detection tests must keep passing (parity for every sniffed type); new registry test asserts sniff -> adapter round-trip for every registered format. Pairs with the loss-modes card (data side vs loss side of the same general-purpose trainer).
