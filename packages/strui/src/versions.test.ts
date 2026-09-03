import { describe, it, expect } from 'vitest'
import {
  STRUI_VERSION,
  FEATURE_VERSIONS,
  getComponentFeature,
  listFeatures,
} from './versions'

// ── STRUI_VERSION ──────────────────────────────────────────────────

describe('STRUI_VERSION', () => {
  it('is a semver string', () => {
    expect(STRUI_VERSION).toMatch(/^\d+\.\d+\.\d+$/)
  })

  it('is currently 0.2.0', () => {
    expect(STRUI_VERSION).toBe('0.2.0')
  })
})

// ── FEATURE_VERSIONS ───────────────────────────────────────────────

describe('FEATURE_VERSIONS', () => {
  it('has core feature', () => {
    expect(FEATURE_VERSIONS.core).toBeDefined()
    expect(FEATURE_VERSIONS.core.backend).toBe('/')
    expect(FEATURE_VERSIONS.core.api).toBe('1.0.0')
  })

  it('has chat feature', () => {
    expect(FEATURE_VERSIONS.chat).toBeDefined()
    expect(FEATURE_VERSIONS.chat.components.length).toBeGreaterThan(0)
  })

  it('has models feature', () => {
    expect(FEATURE_VERSIONS.models).toBeDefined()
    expect(FEATURE_VERSIONS.models.backend).toBe('/models')
  })

  it('has training feature', () => {
    expect(FEATURE_VERSIONS.training).toBeDefined()
    expect(FEATURE_VERSIONS.training.backend).toBe('/training')
  })

  it('has tools feature', () => {
    expect(FEATURE_VERSIONS.tools).toBeDefined()
    expect(FEATURE_VERSIONS.tools.backend).toBe('/agents')
  })

  it('has knowledge feature', () => {
    expect(FEATURE_VERSIONS.knowledge).toBeDefined()
    expect(FEATURE_VERSIONS.knowledge.backend).toBe('/knowledge')
  })

  it('has layout feature', () => {
    expect(FEATURE_VERSIONS.layout).toBeDefined()
    expect(FEATURE_VERSIONS.layout.backend).toBe('/system')
  })

  it('has health feature', () => {
    expect(FEATURE_VERSIONS.health).toBeDefined()
    expect(FEATURE_VERSIONS.health.backend).toBe('/health')
  })

  it('has icons feature', () => {
    expect(FEATURE_VERSIONS.icons).toBeDefined()
    expect(FEATURE_VERSIONS.icons.components.length).toBeGreaterThan(50)
  })

  it('all features have backend field', () => {
    for (const [name, info] of Object.entries(FEATURE_VERSIONS)) {
      expect(info).toHaveProperty('backend')
      expect(typeof info.backend).toBe('string')
    }
  })

  it('all features have api version', () => {
    for (const [name, info] of Object.entries(FEATURE_VERSIONS)) {
      expect(info).toHaveProperty('api')
      expect(info.api).toMatch(/^\d+\.\d+\.\d+$/)
    }
  })

  it('all features have components array', () => {
    for (const [name, info] of Object.entries(FEATURE_VERSIONS)) {
      expect(info).toHaveProperty('components')
      expect(Array.isArray(info.components)).toBe(true)
    }
  })

  it('shared components across features are documented', () => {
    // Some components (StatusDot, StatCard, KpiGrid) appear in multiple features
    // This is intentional — they serve multiple domains
    const shared = ['StatusDot', 'StatCard', 'KpiGrid']
    for (const comp of shared) {
      const occurrences = Object.entries(FEATURE_VERSIONS)
        .filter(([, info]) => info.components.includes(comp))
      expect(occurrences.length).toBeGreaterThanOrEqual(1)
    }
  })
})

// ── getComponentFeature ────────────────────────────────────────────

describe('getComponentFeature', () => {
  it('finds Button in core', () => {
    const result = getComponentFeature('Button')
    expect(result).toEqual({ feature: 'core', version: '1.0.0' })
  })

  it('finds MessageBubble in chat', () => {
    const result = getComponentFeature('MessageBubble')
    expect(result).toEqual({ feature: 'chat', version: '1.0.0' })
  })

  it('finds ModelPicker in models', () => {
    const result = getComponentFeature('ModelPicker')
    expect(result).toEqual({ feature: 'models', version: '1.0.0' })
  })

  it('finds IconSearch in icons', () => {
    const result = getComponentFeature('IconSearch')
    expect(result).toEqual({ feature: 'icons', version: '0.0.0' })
  })

  it('finds AppShell in layout', () => {
    const result = getComponentFeature('AppShell')
    expect(result).toEqual({ feature: 'layout', version: '1.0.0' })
  })

  it('returns null for unknown component', () => {
    expect(getComponentFeature('UnknownWidget')).toBeNull()
  })

  it('returns null for empty string', () => {
    expect(getComponentFeature('')).toBeNull()
  })
})

// ── listFeatures ───────────────────────────────────────────────────

describe('listFeatures', () => {
  it('returns array of features', () => {
    const features = listFeatures()
    expect(Array.isArray(features)).toBe(true)
    expect(features.length).toBeGreaterThan(0)
  })

  it('each entry has required fields', () => {
    const features = listFeatures()
    for (const f of features) {
      expect(f).toHaveProperty('name')
      expect(f).toHaveProperty('backend')
      expect(f).toHaveProperty('api')
      expect(f).toHaveProperty('componentCount')
      expect(typeof f.name).toBe('string')
      expect(typeof f.backend).toBe('string')
      expect(typeof f.api).toBe('string')
      expect(typeof f.componentCount).toBe('number')
    }
  })

  it('includes core feature', () => {
    const features = listFeatures()
    const core = features.find(f => f.name === 'core')
    expect(core).toBeDefined()
    expect(core!.backend).toBe('/')
  })

  it('componentCount matches actual components', () => {
    const features = listFeatures()
    for (const f of features) {
      const actual = FEATURE_VERSIONS[f.name as keyof typeof FEATURE_VERSIONS]
      expect(f.componentCount).toBe(actual.components.length)
    }
  })

  it('returns all features', () => {
    expect(listFeatures().length).toBe(Object.keys(FEATURE_VERSIONS).length)
  })
})
