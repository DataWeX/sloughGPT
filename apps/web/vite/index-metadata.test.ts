/**
 * Drift guard: lib/document-meta.ts must match apps/web/index.html
 * (Vite's document shell). Fails if either side changes without the other.
 */
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { metadata, viewport } from '../lib/document-meta'

const webRoot = join(import.meta.dirname, '..')
const html = readFileSync(join(webRoot, 'index.html'), 'utf-8')

describe('index.html ↔ document-meta parity', () => {
  it('title matches', () => {
    expect(html).toContain(`<title>${metadata.title}</title>`)
  })

  it('description matches', () => {
    expect(html).toContain(`name="description" content="${metadata.description}"`)
  })

  it('favicon matches icons.icon', () => {
    expect(html).toContain(`rel="icon" href="${metadata.icons.icon}"`)
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
