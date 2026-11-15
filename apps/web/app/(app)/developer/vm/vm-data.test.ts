// Pure-function tests for the VM training config boundary. Both the launch
// path (buildTrainSource) and the persistence path (useVmPage save/load)
// round-trip through clampTrainConfig, so out-of-range values must be
// clamped here — while NaN (cleared/invalid input) keeps falling back to
// the defaults, per the hint contract asserted in vm-tab.test.tsx.
import { describe, it, expect } from 'vitest'
import { clampTrainConfig, DEFAULT_TRAIN_CONFIG } from './vm-data'

describe('clampTrainConfig', () => {
  it('clamps out-of-range values to the shared training limits', () => {
    expect(
      clampTrainConfig({
        dataset: 'shakespeare',
        epochs: 99999,
        lr: 5,
        batch_size: 99999,
        n_layer: 999,
        n_head: 999,
        embed_dim: 99999,
      }),
    ).toEqual({
      ...DEFAULT_TRAIN_CONFIG,
      epochs: 1000,
      lr: 1,
      batch_size: 512,
      n_layer: 24,
      n_head: 64,
      embed_dim: 1024,
    })
  })

  it('falls back to defaults for NaN or non-positive fields', () => {
    expect(
      clampTrainConfig({
        dataset: '',
        epochs: NaN,
        lr: NaN,
        batch_size: NaN,
        n_layer: NaN,
        n_head: NaN,
        embed_dim: NaN,
      }),
    ).toEqual(DEFAULT_TRAIN_CONFIG)

    expect(clampTrainConfig({ ...DEFAULT_TRAIN_CONFIG, lr: 0 }).lr).toBe(DEFAULT_TRAIN_CONFIG.lr)
  })

  it('keeps an in-range config unchanged', () => {
    expect(clampTrainConfig({ ...DEFAULT_TRAIN_CONFIG })).toEqual(DEFAULT_TRAIN_CONFIG)
  })
})
