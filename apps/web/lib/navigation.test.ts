import { describe, it, expect } from 'vitest'
import { NAV_SECTIONS, SHORTCUT_TO_PATH, ALL_ROUTES, SIDEBAR_ICONS, SIDEBAR_SECTIONS, NAV_ICONS } from './navigation'

describe('navigation', () => {
  it('has 4 sections', () => {
    expect(NAV_SECTIONS).toHaveLength(5)
  })

  it('each section has a labelKey and at least one route', () => {
    for (const section of NAV_SECTIONS) {
      expect(section.labelKey).toMatch(/^nav\.section\./)
      expect(section.routes.length).toBeGreaterThanOrEqual(1)
    }
  })

  it('every route has a unique path', () => {
    const paths = ALL_ROUTES.map(r => r.path)
    expect(new Set(paths).size).toBe(paths.length)
  })

  it('every route has a labelKey', () => {
    for (const route of ALL_ROUTES) {
      expect(route.labelKey).toMatch(/^nav\./)
    }
  })

  it('SHORTCUT_TO_PATH maps shortcuts to correct paths', () => {
    expect(SHORTCUT_TO_PATH['1']).toBe('/chat')
    expect(SHORTCUT_TO_PATH['4']).toBe('/models')
    expect(SHORTCUT_TO_PATH['6']).toBe('/souls')
    expect(SHORTCUT_TO_PATH['shift+A']).toBe('/settings')
  })

  it('every shortcut key maps to a valid route path', () => {
    const routePaths = new Set(ALL_ROUTES.map(r => r.path))
    for (const [key, path] of Object.entries(SHORTCUT_TO_PATH)) {
      expect(routePaths.has(path)).toBe(true)
    }
  })

  it('SIDEBAR_ICONS has an entry for every sidebar route', () => {
    for (const route of ALL_ROUTES) {
      expect(SIDEBAR_ICONS[route.path]).toBeDefined()
    }
  })

  it('ALL_ROUTES is flat concatenation of sections', () => {
    const flat = NAV_SECTIONS.flatMap(s => s.routes)
    expect(ALL_ROUTES).toEqual(flat)
  })

  it('SIDEBAR_SECTIONS hides folded routes but keeps them in ALL_ROUTES', () => {
    const sidebarPaths = new Set(SIDEBAR_SECTIONS.flatMap(s => s.routes.map(r => r.path)))
    for (const folded of ['/training/grid-search', '/training/presets', '/training/runs']) {
      expect(sidebarPaths.has(folded)).toBe(false)
      expect(ALL_ROUTES.some(r => r.path === folded)).toBe(true)
    }
    expect(sidebarPaths.has('/training')).toBe(true)
  })

  it('every sidebar route resolves an icon via NAV_ICONS', () => {
    for (const section of SIDEBAR_SECTIONS) {
      for (const route of section.routes) {
        expect(route.icon).toBeDefined()
        expect(NAV_ICONS[route.icon!]).toBeDefined()
      }
    }
  })

  it('zap/compare/file aliases resolve to their own icons, not duplicates', async () => {
    const { IconBolt, IconChart, IconCompare, IconDocument, IconFile, IconSparkle } = await import('@/components/icons/NavIcons')
    expect(NAV_ICONS.zap).toBe(IconBolt)
    expect(NAV_ICONS.compare).toBe(IconCompare)
    expect(NAV_ICONS.file).toBe(IconFile)
    expect(NAV_ICONS.zap).not.toBe(IconSparkle)
    expect(NAV_ICONS.compare).not.toBe(IconChart)
    expect(NAV_ICONS.file).not.toBe(IconDocument)
  })

  it('core sidebar routes each use a distinct icon', () => {
    const core = SIDEBAR_SECTIONS.find(s => s.labelKey === 'nav.section.core')
    expect(core).toBeDefined()
    const icons = core!.routes.map(r => r.icon)
    expect(new Set(icons).size).toBe(icons.length)
  })
})
