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

  it('manifest link matches metadata.manifest and manifest.json agrees', () => {
    expect(html).toContain(`<link rel="manifest" href="${metadata.manifest}" />`)
    const manifest = JSON.parse(readFileSync(join(webRoot, 'public', 'manifest.json'), 'utf-8'))
    expect(manifest.name).toBe(metadata.title)
    expect(manifest.short_name).toBe('Man')
    expect(manifest.description).toBe(metadata.description)
  })

  it('favicon matches icons.icon', () => {
    expect(html).toContain(`rel="icon" href="${metadata.icons.icon}"`)
  })

  it('Open Graph + Twitter tags match document-meta and og.png exists', () => {
    const og = metadata.openGraph
    expect(html).toContain(`property="og:title" content="${og.title}"`)
    expect(html).toContain(`property="og:description" content="${og.description}"`)
    expect(html).toContain(`property="og:type" content="${og.type}"`)
    expect(html).toContain(`property="og:site_name" content="${og.siteName}"`)
    expect(html).toContain(`property="og:locale" content="${og.locale}"`)
    expect(html).toContain(`property="og:image" content="${og.image}"`)
    expect(html).toContain(`property="og:image:width" content="${og.imageWidth}"`)
    expect(html).toContain(`property="og:image:height" content="${og.imageHeight}"`)
    expect(html).toContain(`property="og:image:alt" content="${og.imageAlt}"`)
    const tw = metadata.twitter
    expect(html).toContain(`name="twitter:card" content="${tw.card}"`)
    expect(html).toContain(`name="twitter:title" content="${tw.title}"`)
    expect(html).toContain(`name="twitter:description" content="${tw.description}"`)
    expect(html).toContain(`name="twitter:image" content="${tw.image}"`)
    expect(og.title).toBe(metadata.title)
    expect(og.description).toBe(metadata.description)
    expect(tw.title).toBe(metadata.title)
    expect(() => readFileSync(join(webRoot, 'public', 'og.png'))).not.toThrow()
    expect(html).not.toMatch(/property="og:url"/)
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
