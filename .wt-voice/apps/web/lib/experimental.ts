/**
 * Web-side tracking point for experimental features.
 *
 * The strui registry (FEATURE_VERSIONS) no longer flags experimental or
 * freshness status, so the web app owns the list. Components check these
 * helpers instead of hardcoding status.
 */

/** Feature keys flagged experimental in the registry. */
const EXPERIMENTAL_FEATURES: ReadonlySet<string> = new Set([
  'consciousness',
  'voice',
  'phoneme',
  'tokenizer',
])

/** Features flagged experimental in the registry. */
export function experimentalFeatures(): string[] {
  return [...EXPERIMENTAL_FEATURES]
}

/** True when the feature is tracked as experimental. */
export function isTrackedExperimental(feature: string): boolean {
  return EXPERIMENTAL_FEATURES.has(feature)
}

/**
 * True when the feature is inside its freshness window.
 * All tracked experimental surfaces are new, so freshness currently
 * mirrors tracked status. Unknown features are never fresh.
 */
export function isFresh(feature: string): boolean {
  return EXPERIMENTAL_FEATURES.has(feature)
}
