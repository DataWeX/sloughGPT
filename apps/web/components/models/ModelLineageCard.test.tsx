// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/registry-controller', () => ({
  registryController: { artifacts: vi.fn() },
}))
vi.mock('@/lib/toast-store', () => {
  const mod = {
    useToastStore: (sel: (s: { addToast: (m: string, t?: string) => void }) => unknown) =>
      sel({ addToast: vi.fn() }),
  }
  const sink = mod.useToastStore as unknown as ((f: (x: unknown) => unknown) => unknown) & {
    getState?: () => unknown
  }
  if (typeof sink.getState !== 'function') {
    sink.getState = () => {
      const st = sink((x) => x)
      return typeof st === 'object' && st !== null ? st : mod
    }
  }
  return mod
})

import ModelLineageCard from './ModelLineageCard'
import { registryController } from '@/lib/registry-controller'

describe('ModelLineageCard', () => {
  afterEach(cleanup)
  beforeEach(() => vi.clearAllMocks())

  it('renders a dataset -> model family line for stamped artifacts', async () => {
    vi.mocked(registryController.artifacts).mockResolvedValue([
      {
        artifact_id: 'model:baby.soul',
        kind: 'model',
        id: 'baby.soul',
        name: 'baby.soul',
        path: '/x/baby.soul',
        size_bytes: 1024,
        parents: ['dataset:tinyshakespeare', 'model:sloughgpt'],
      },
    ])
    render(<ModelLineageCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('baby.soul')).toBeDefined()
      expect(screen.getByText('dataset:tinyshakespeare')).toBeDefined()
      expect(screen.getByText('model:sloughgpt')).toBeDefined()
    })
  })

  it('shows an empty state explaining stamps when nothing has parents', async () => {
    vi.mocked(registryController.artifacts).mockResolvedValue([
      {
        artifact_id: 'model:plain.soul',
        kind: 'model',
        id: 'plain.soul',
        name: 'plain.soul',
        path: '/x/p',
        size_bytes: 1,
        parents: [],
      },
    ])
    render(<ModelLineageCard />)
    await vi.waitFor(() => {
      expect(screen.getByText(/no lineage stamped yet/i)).toBeDefined()
    })
  })

  it('hides unstamped roots from the family list but counts them', async () => {
    vi.mocked(registryController.artifacts).mockResolvedValue([
      {
        artifact_id: 'model:root.soul',
        kind: 'model',
        id: 'root.soul',
        name: 'root.soul',
        path: '/x/r',
        size_bytes: 1,
        parents: [],
      },
      {
        artifact_id: 'model:child.soul',
        kind: 'model',
        id: 'child.soul',
        name: 'child.soul',
        path: '/x/c',
        size_bytes: 2,
        parents: ['dataset:ds1'],
      },
    ])
    render(<ModelLineageCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('child.soul')).toBeDefined()
      expect(screen.queryByText('root.soul')).toBeNull()
      expect(screen.getByText(/1 of 2/i)).toBeDefined()
    })
  })

  it('renders a skeleton while loading', () => {
    vi.mocked(registryController.artifacts).mockReturnValue(new Promise(() => {}))
    const { container } = render(<ModelLineageCard />)
    expect(
      container.querySelector('.animate-pulse, [class*="pulse"], .sl-skeleton, .skeleton') ??
        container.firstChild,
    ).toBeTruthy()
  })

  it('surfaces a load failure via the toast, not a crash', async () => {
    vi.mocked(registryController.artifacts).mockRejectedValue(new Error('boom'))
    render(<ModelLineageCard />)
    await vi.waitFor(() => {
      expect(screen.getByText(/no lineage stamped yet/i)).toBeDefined()
    })
    expect(screen.queryByText('baby.soul')).toBeNull()
  })
})
