/**
 * Drift guard: app/layout.tsx metadata/viewport must match apps/web/index.html
 * (Vite's document shell). Fails if either side changes without the other.
 */
import { describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

vi.mock('next/font/local', () => ({
  default: () => ({ variable: '--mock-font' }),
}))

const { metadata, viewport } = await import('../app/layout')

const webRoot = join(import.meta.dirname, '..')
const html = readFileSync(join(webRoot, 'index.html'), 'utf-8')

describe('index.html ↔ layout metadata parity', () => {
  it('title matches', () => {
    expect(html).toContain(`<title>${metadata.title}</title>`)
  })

  it('description matches', () => {
    expect(html).toContain(`name="description" content="${metadata.description}"`)
  })

  it('favicon matches icons.icon', () => {
    expect(metadata.icons).toBeTruthy()
    const icon = (metadata.icons as { icon?: string }).icon
    expect(html).toContain(`rel="icon" href="${icon}"`)
  })

  it('viewport meta matches viewport export', () => {
    expect(viewport.width).toBe('device-width')
    expect(viewport.initialScale).toBe(1)
    expect(viewport.viewportFit).toBe('cover')
    expect(html).toContain('name="viewport"')
    expect(html).toContain('width=device-width')
    expect(html).toContain('viewport-fit=cover')
  })

  it('html lang is en', () => {
    expect(html).toContain('<html lang="en"')
  })

  it('fonts preloaded and CSS vars present', () => {
    expect(html).toContain('rel="preload"')
    expect(html).toContain('/fonts/outfit-latin.woff2')
    expect(html).toContain('/fonts/jetbrains-latin.woff2')
    expect(html).toContain('--font-rubik')
    expect(html).toContain('--font-jetbrains-mono')
  })
})
