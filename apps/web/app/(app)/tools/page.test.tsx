import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { ToolProfile } from '@/lib/tools-controller'

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

const { mockListTools } = vi.hoisted(() => ({
  mockListTools: vi.fn(),
}))

vi.mock('@/lib/tools-controller', () => ({
  listTools: mockListTools,
}))

import ToolsPage from './page'

function makeTool(id: string, name: string, description: string, icon: string): ToolProfile {
  return {
    id,
    name,
    description,
    icon,
    params: [],
    options: {},
    default_options: {},
    system_prompt: '',
    max_tokens: 500,
  }
}

const TOOLS = [
  makeTool('writing', 'Writing Assistant', 'Write prose', 'document'),
  makeTool('translate', 'Translate', 'Translate text', 'chat'),
  makeTool('brainstorm', 'Brainstorm', 'Ideas fast', 'brain'),
]

describe('ToolsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockListTools.mockResolvedValue(TOOLS)
  })

  it('renders the page title', async () => {
    render(<ToolsPage />)
    expect(screen.getByText('Tools')).toBeDefined()
    await screen.findByText('Writing Assistant')
    expect(mockListTools).toHaveBeenCalledTimes(1)
  })

  it('renders a card for each tool from the backend', async () => {
    render(<ToolsPage />)
    expect(await screen.findByText('Writing Assistant')).toBeDefined()
    expect(screen.getByText('Write prose')).toBeDefined()
    expect(screen.getByText('Translate')).toBeDefined()
    expect(screen.getByText('Translate text')).toBeDefined()
    expect(screen.getByText('Ideas fast')).toBeDefined()
  })

  it('links each tool card to its route', async () => {
    render(<ToolsPage />)
    const writingLink = await screen.findByRole('link', { name: /Writing Assistant/ })
    expect(writingLink).toHaveAttribute('href', '/writing')
    expect(screen.getByRole('link', { name: /Translate/ })).toHaveAttribute('href', '/translate')
    expect(screen.getByRole('link', { name: /Brainstorm/ })).toHaveAttribute('href', '/brainstorm')
  })

  it('shows an empty state when no tools are returned', async () => {
    mockListTools.mockResolvedValue([])
    render(<ToolsPage />)
    expect(await screen.findByText(/No tools yet/i)).toBeDefined()
  })

  it('shows a loading placeholder while fetching', () => {
    mockListTools.mockReturnValue(new Promise(() => {}))
    render(<ToolsPage />)
    expect(screen.getByText(/Loading tools/i)).toBeDefined()
  })
})