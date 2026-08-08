import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

vi.mock('@/lib/agents-controller', () => ({
  agentsController: {
    list: vi.fn().mockResolvedValue([]),
    create: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
    execute: vi.fn(),
    orchestrate: vi.fn(),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-08-07',
}))

vi.mock('@/lib/validation-schemas', () => ({
  agentSchema: { shape: {} },
  agentExecuteSchema: { shape: {} },
  orchestrateSchema: { shape: {} },
}))

import AgentsPage from './page'

describe('AgentsPage', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(() => { cleanup() })

  it('renders without crashing', async () => {
    render(<AgentsPage />)
    expect(document.body).toBeTruthy()
    await screen.findAllByText(/no|empty|nothing|0/i)
  })

  it('shows empty state when no agents', async () => {
    render(<AgentsPage />)
    await screen.findAllByText(/no|empty|nothing|0/i)
  })
})
