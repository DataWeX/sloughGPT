// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { QuickImportSection, type QuickImportSource } from './quick-import-section'

function makeSources(overrides: Partial<QuickImportSource> = {}): QuickImportSource[] {
  return [
    {
      key: 'kaggle',
      label: 'Kaggle',
      icon: null,
      hint: 'Requires Kaggle API credentials configured on the server.',
      presets: [
        { id: 'heptapod/titanic', label: 'Titanic' },
        { id: 'uciml/iris', label: 'Iris' },
      ],
      importingId: null,
      onImport: vi.fn(),
      ...overrides,
    },
    {
      key: 'huggingface',
      label: 'HuggingFace',
      icon: null,
      hint: 'No server setup — pulls straight from the HuggingFace Hub.',
      presets: [{ id: 'HuggingFaceH4/tinyshakespeare', label: 'Tiny Shakespeare' }],
      importingId: null,
      onImport: vi.fn(),
    },
  ]
}

describe('QuickImportSection', () => {
  afterEach(cleanup)

  it('renders a heading badge listing every source', () => {
    render(<QuickImportSection sources={makeSources()} />)
    expect(screen.getByText('Quick Import')).toBeDefined()
    expect(screen.getByText('Kaggle · HuggingFace')).toBeDefined()
  })

  it('renders each source group with its presets and requirement note', () => {
    render(<QuickImportSection sources={makeSources()} />)
    expect(screen.getByRole('region', { name: 'Kaggle presets' })).toBeDefined()
    expect(screen.getByRole('region', { name: 'HuggingFace presets' })).toBeDefined()
    expect(screen.getByText('Titanic')).toBeDefined()
    expect(screen.getByText('Iris')).toBeDefined()
    expect(screen.getByText('Tiny Shakespeare')).toBeDefined()
    expect(
      screen.getByText('Requires Kaggle API credentials configured on the server.'),
    ).toBeDefined()
    expect(
      screen.getByText('No server setup — pulls straight from the HuggingFace Hub.'),
    ).toBeDefined()
  })

  it('counts presets per source', () => {
    render(<QuickImportSection sources={makeSources()} />)
    expect(screen.getByText('2 presets')).toBeDefined()
    expect(screen.getByText('1 preset')).toBeDefined()
  })

  it('calls onImport with the preset id', () => {
    const sources = makeSources()
    render(<QuickImportSection sources={sources} />)
    fireEvent.click(screen.getByText('Titanic'))
    expect(sources[0].onImport).toHaveBeenCalledWith('heptapod/titanic')
  })

  it('marks the in-flight preset as loading and blocks the rest', () => {
    const sources = makeSources({ importingId: 'uciml/iris' })
    render(<QuickImportSection sources={sources} />)

    const loading = screen.getByText('Importing...').closest('button')!
    expect(loading).toHaveProperty('disabled', true)

    const idle = screen.getByText('Titanic').closest('button')!
    expect(idle).toHaveProperty('disabled', true)

    const otherSource = screen.getByText('Tiny Shakespeare').closest('button')!
    expect(otherSource).toHaveProperty('disabled', false)
  })

  it('renders folded content regardless of the open state', () => {
    render(<QuickImportSection sources={makeSources()} defaultOpen />)
    const details = document.querySelector('details')!
    expect(details).toHaveProperty('open', true)
    expect(screen.getByText('Titanic')).toBeDefined()
  })
})
