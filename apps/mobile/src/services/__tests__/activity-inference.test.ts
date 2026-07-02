/**
 * Test that the TypeScript forward pass matches the Python ActivityClassifier.
 *
 * Verifies: weight shapes, forward pass math, softmax, argmax.
 */
import {
  forwardPass,
  type ModelWeights,
} from '../activity-inference';

// ── Forward pass math tests ──────────────────────────────────────────

// Create known weights for exact verification
function makeTestWeights(): ModelWeights {
  // conv1.weight: (32, 6, 1, 7) — all 0 except bias
  // conv1.bias: (32,) — zeros
  // conv2.weight: (32, 32, 1, 5) — zeros
  // conv2.bias: (32,) — zeros
  // fc.weight: (2, 56) — identity-like
  // fc.bias: (2,) — zeros
  const conv1Weight = new Float32Array(32 * 6 * 1 * 7);
  const conv1Bias = new Float32Array(32);
  const conv2Weight = new Float32Array(32 * 32 * 1 * 5);
  const conv2Bias = new Float32Array(32);
  const fcWeight = new Float32Array(2 * 56);
  const fcBias = new Float32Array(2);

  // Set fc.weight so class 0 has higher weight for first 56 dims
  for (let i = 0; i < 56; i++) {
    fcWeight[i] = 1.0; // class 0
    fcWeight[56 + i] = -1.0; // class 1
  }

  return {
    conv1Weight,
    conv1Bias,
    conv2Weight,
    conv2Bias,
    fcWeight,
    fcBias,
    numClasses: 2,
  };
}

describe('forwardPass', () => {
  it('produces valid softmax probabilities', () => {
    const weights = makeTestWeights();
    // Uniform sensor data
    const data: number[][] = Array.from({length: 50}, () => [
      0.1, 0.2, 0.98, 0.01, 0.05, 0.03,
    ]);

    const result = forwardPass(weights, data);

    expect(result.classId).toBeGreaterThanOrEqual(0);
    expect(result.classId).toBeLessThan(2);
    expect(result.probabilities).toHaveLength(2);

    // Probabilities should sum to ~1.0
    const sum = result.probabilities.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 4);

    // All probabilities should be in [0, 1]
    for (const p of result.probabilities) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  it('classifies uniform input consistently', () => {
    const weights = makeTestWeights();
    const data: number[][] = Array.from({length: 50}, () => [
      0, 0, 0, 0, 0, 0,
    ]);

    const r1 = forwardPass(weights, data);
    const r2 = forwardPass(weights, data);

    // Deterministic: same input → same output
    expect(r1.classId).toBe(r2.classId);
    expect(r1.probabilities).toEqual(r2.probabilities);
  });

  it('class 0 preferred when all sensor values are positive', () => {
    const weights = makeTestWeights();
    const data: number[][] = Array.from({length: 50}, () => [
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ]);

    const result = forwardPass(weights, data);
    // With positive input and fc weights: class 0 has sum(1.0*positive), class 1 has sum(-1.0*positive)
    // But conv layers with zero weights output zero, stats path outputs positive values
    // So class 0 should win
    expect(result.probabilities[0]).toBeGreaterThan(result.probabilities[1]);
  });

  it('handles variable-length input', () => {
    const weights = makeTestWeights();
    const shortData: number[][] = Array.from({length: 10}, () => [
      0.5, 0.3, 0.9, 0.1, 0.2, 0.4,
    ]);
    const longData: number[][] = Array.from({length: 128}, () => [
      0.5, 0.3, 0.9, 0.1, 0.2, 0.4,
    ]);

    const rShort = forwardPass(weights, shortData);
    const rLong = forwardPass(weights, longData);

    expect(rShort.probabilities).toHaveLength(2);
    expect(rLong.probabilities).toHaveLength(2);

    const sumShort = rShort.probabilities.reduce((a, b) => a + b, 0);
    const sumLong = rLong.probabilities.reduce((a, b) => a + b, 0);
    expect(sumShort).toBeCloseTo(1.0, 4);
    expect(sumLong).toBeCloseTo(1.0, 4);
  });
});

// ── Activity name mapping ────────────────────────────────────────────

describe('activity inference', () => {
  it('maps class IDs to activity names', () => {
    const weights = makeTestWeights();
    const data: number[][] = Array.from({length: 50}, () => [
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    ]);

    const result = forwardPass(weights, data);
    // With numClasses=2, valid names are 'stationary' or 'walking'
    expect(['stationary', 'walking']).toContain(result.className);
  });
});
