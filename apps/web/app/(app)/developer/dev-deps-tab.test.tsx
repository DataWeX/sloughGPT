import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { DevDepsTab } from './dev-deps-tab'

// The artifact is build-time generated; the tab must render a fixture, never
// the live file, so grouping assertions stay deterministic.
vi.mock('@/lib/deps.gen', () => ({
  DEPS: {
    manifests: ['apps/web/package.json', 'package.json', 'pyproject.toml', 'requirements.txt'],
    entries: [
      { name: 'numpy', key: 'numpy', version: '>=1.24.0', tree: 'python', kind: 'runtime', manifest: 'pyproject.toml' },
      { name: 'numpy', key: 'numpy', version: '>=1.26.0', tree: 'python', kind: 'runtime', manifest: 'requirements.txt' },
      { name: 'requests', key: 'requests', version: '>=2.28.0', tree: 'python', kind: 'runtime', manifest: 'requirements.txt' },
      { name: 'react', key: 'react', version: '^19.0.0', tree: 'js', kind: 'runtime', manifest: 'apps/web/package.json' },
      { name: 'react', key: 'react', version: '^18.2.0', tree: 'js', kind: 'dev', manifest: 'package.json' },
    ],
    duplicates: [
      {
        key: 'numpy',
        entries: [
          { name: 'numpy', key: 'numpy', version: '>=1.24.0', tree: 'python', kind: 'runtime', manifest: 'pyproject.toml' },
          { name: 'numpy', key: 'numpy', version: '>=1.26.0', tree: 'python', kind: 'runtime', manifest: 'requirements.txt' },
        ],
        versions: ['>=1.24.0', '>=1.26.0'],
      },
      {
        key: 'react',
        entries: [
          { name: 'react', key: 'react', version: '^18.2.0', tree: 'js', kind: 'dev', manifest: 'package.json' },
          { name: 'react', key: 'react', version: '^19.0.0', tree: 'js', kind: 'runtime', manifest: 'apps/web/package.json' },
        ],
        versions: ['^18.2.0', '^19.0.0'],
      },
    ],
    drift: [
      {
        key: 'numpy',
        entries: [
          { name: 'numpy', key: 'numpy', version: '>=1.24.0', tree: 'python', kind: 'runtime', manifest: 'pyproject.toml' },
          { name: 'numpy', key: 'numpy', version: '>=1.26.0', tree: 'python', kind: 'runtime', manifest: 'requirements.txt' },
        ],
        versions: ['>=1.24.0', '>=1.26.0'],
      },
      {
        key: 'react',
        entries: [
          { name: 'react', key: 'react', version: '^18.2.0', tree: 'js', kind: 'dev', manifest: 'package.json' },
          { name: 'react', key: 'react', version: '^19.0.0', tree: 'js', kind: 'runtime', manifest: 'apps/web/package.json' },
        ],
        versions: ['^18.2.0', '^19.0.0'],
      },
    ],
    cross_tree: ['requests'],
  },
}))

describe('DevDepsTab', () => {
  afterEach(cleanup)

  it('summarizes unique packages per tree, manifest count, and drift count', () => {
    render(<DevDepsTab />)
    // unique keys: python = numpy + requests, js = react
    expect(screen.getByText('Python packages')).toBeDefined()
    expect(screen.getByText('JavaScript packages')).toBeDefined()
    expect(screen.getByText('Manifests')).toBeDefined()
    expect(screen.getByText('Drift')).toBeDefined()
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(2) // python unique + drift groups
    expect(screen.getByText('4')).toBeDefined() // manifests
    expect(screen.getByText('1')).toBeDefined() // js unique
  })

  it('makes drift visually obvious: package with every declared version', () => {
    render(<DevDepsTab />)
    expect(screen.getAllByText('numpy').length).toBeGreaterThan(0)
    expect(screen.getAllByText('react').length).toBeGreaterThan(0)
    expect(screen.getAllByText('>=1.24.0').length).toBeGreaterThan(0)
    expect(screen.getAllByText('>=1.26.0').length).toBeGreaterThan(0)
    expect(screen.getAllByText('drift').length).toBeGreaterThan(0)
  })

  it('highlights packages declared by both trees', () => {
    render(<DevDepsTab />)
    // 'requests' also appears in the collapsed python inventory — any occurrence
    expect(screen.getAllByText('requests').length).toBeGreaterThan(0)
    expect(screen.getByText(/both trees/i)).toBeDefined()
  })

  it('keeps the raw inventories collapsed by default', () => {
    const { container } = render(<DevDepsTab />)
    const folds = Array.from(container.querySelectorAll('details'))
    expect(folds.length).toBeGreaterThanOrEqual(3) // duplicates + python + js inventories
    expect(folds.every((d) => !d.open)).toBe(true)
    // collapsed ≠ gone: the raw entries are still in the DOM, one click away
    expect(folds.map((d) => d.textContent).join('')).toContain('pyproject.toml')
  })

  it('renders without fetching — the artifact is build-time, no endpoint earned', () => {
    const fetchSpy = vi.fn()
    const original = globalThis.fetch
    globalThis.fetch = fetchSpy as unknown as typeof fetch
    render(<DevDepsTab />)
    expect(fetchSpy).not.toHaveBeenCalled()
    globalThis.fetch = original
  })
})
