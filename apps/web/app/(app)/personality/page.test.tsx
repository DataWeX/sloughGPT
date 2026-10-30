import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act, waitFor, fireEvent } from '@testing-library/react'
import PersonalityPage from './page'

const {
  mockGetPersonality,
  mockGetPersonalityPresets,
  mockGetPersonalityHistory,
  mockGetPersonalityConflicts,
  mockListPersonas,
  mockResetPersonality,
  mockApplyPersonalityPreset,
  mockSavePersona,
  mockActivatePersona,
  mockDeletePersona,
  mockApiPatch,
  mockAddToast,
} = vi.hoisted(() => ({
  mockGetPersonality: vi.fn(),
  mockGetPersonalityPresets: vi.fn(),
  mockGetPersonalityHistory: vi.fn(),
  mockGetPersonalityConflicts: vi.fn(),
  mockListPersonas: vi.fn(),
  mockResetPersonality: vi.fn(),
  mockApplyPersonalityPreset: vi.fn(),
  mockSavePersona: vi.fn(),
  mockActivatePersona: vi.fn(),
  mockDeletePersona: vi.fn(),
  mockApiPatch: vi.fn(),
  mockAddToast: vi.fn(),
}))

const mockProfile = {
  values: ['honesty', 'curiosity'],
  goals: ['be helpful'],
  voice: { formality: 0.5, warmth: 0.7, confidence: 0.6, humor: 0.3, verbosity: 0.4, empathy: 0.8 },
  style: {
    use_examples: true,
    ask_follow_ups: false,
    acknowledge_uncertainty: true,
    use_analogies: false,
    break_down_complex_topics: true,
  },
  traits: {
    openness: 0.6,
    conscientiousness: 0.7,
    extraversion: 0.4,
    agreeableness: 0.8,
    neuroticism: 0.3,
  },
  interests: ['AI', 'science'],
  avoid: ['rudeness'],
}

vi.mock('@/lib/config', () => ({ PUBLIC_API_URL: 'http://localhost:8000' }))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: typeof mockAddToast }) => typeof mockAddToast) =>
    selector({ addToast: mockAddToast }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (key: string) => key, locale: 'en' }),
}))

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title }: { children: React.ReactNode; title: string }) => (
    <div data-testid="page-container" data-title={title}>
      {children}
    </div>
  ),
}))

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Button: ({ children, onClick, disabled, variant }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant}>
        {children}
      </button>
    ),
    Card: passthrough,
    CardContent: passthrough,
    CardDescription: ({ children }: any) => <div>{children}</div>,
    CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Input: ({ value, onChange, placeholder, ...props }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} {...props} />
    ),
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    IconHeart: ({ filled }: { filled?: boolean }) => (
      <span data-testid="icon-heart" data-filled={String(!!filled)} />
    ),
  }
})

vi.mock('@/lib/http-client', () => ({
  apiPatch: mockApiPatch,
}))

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getPersonality: mockGetPersonality,
    getPersonalityPresets: mockGetPersonalityPresets,
    getPersonalityHistory: mockGetPersonalityHistory,
    getPersonalityConflicts: mockGetPersonalityConflicts,
    listPersonas: mockListPersonas,
    resetPersonality: mockResetPersonality,
    applyPersonalityPreset: mockApplyPersonalityPreset,
    savePersona: mockSavePersona,
    activatePersona: mockActivatePersona,
    deletePersona: mockDeletePersona,
  },
}))

const consciousnessController = {
  getPersonality: mockGetPersonality,
  getPersonalityPresets: mockGetPersonalityPresets,
  getPersonalityHistory: mockGetPersonalityHistory,
  getPersonalityConflicts: mockGetPersonalityConflicts,
  listPersonas: mockListPersonas,
  resetPersonality: mockResetPersonality,
  applyPersonalityPreset: mockApplyPersonalityPreset,
  savePersona: mockSavePersona,
  activatePersona: mockActivatePersona,
  deletePersona: mockDeletePersona,
}

const apiPatch = mockApiPatch

beforeEach(() => {
  mockGetPersonality.mockResolvedValue(mockProfile)
  mockGetPersonalityPresets.mockResolvedValue({ names: ['default', 'formal', 'creative'] })
  mockGetPersonalityHistory.mockResolvedValue({
    history: [
      { timestamp: Date.now() / 1000 - 3600, voice: { warmth: 0.5 }, traits: { openness: 0.4 } },
      { timestamp: Date.now() / 1000, voice: { warmth: 0.7 }, traits: { openness: 0.6 } },
    ],
  })
  mockGetPersonalityConflicts.mockResolvedValue({
    conflicts: [
      { type: 'style', severity: 'medium', message: 'Conflict', fields: ['humor', 'formality'] },
    ],
  })
  mockListPersonas.mockResolvedValue({
    personas: [
      {
        id: 'p1',
        name: 'Test Persona',
        values: ['honesty'],
        traits: { openness: 0.8, warmth: 0.7, formality: 0.2 },
        top_traits: [
          { key: 'openness', value: 0.8 },
          { key: 'warmth', value: 0.7 },
        ],
        active: false,
      },
      { id: 'p2', name: 'Active Persona', values: ['kindness'], active: true },
    ],
  })
  mockResetPersonality.mockResolvedValue({ reset: true })
  mockApplyPersonalityPreset.mockResolvedValue({ applied: true })
  mockSavePersona.mockResolvedValue({ id: 'new-p' })
  mockActivatePersona.mockResolvedValue({ activated: true })
  mockDeletePersona.mockResolvedValue({ deleted: true })
  mockApiPatch.mockResolvedValue(mockProfile)
})

afterEach(() => {
  vi.restoreAllMocks()
})

async function waitForLoad() {
  await waitFor(() => {
    expect(screen.getByText('Core Identity')).toBeInTheDocument()
  })
}

describe('PersonalityPage', () => {
  it('shows page container with correct title', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    expect(screen.getByTestId('page-container')).toHaveAttribute('data-title', 'Personality')
  })

  it('fetches profile on mount and displays sections', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    expect(screen.getByText('Core Identity')).toBeInTheDocument()
    expect(screen.getByText('Voice')).toBeInTheDocument()
    expect(screen.getByText('Traits')).toBeInTheDocument()
    expect(screen.getByText('Communication Style')).toBeInTheDocument()
  })

  it('displays saved personas', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    expect(screen.getByText('Test Persona')).toBeInTheDocument()
  })

  it('shows the rose-blue traits icon on persona cards', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    const icons = screen.getAllByTestId('icon-heart')
    expect(icons.length).toBeGreaterThanOrEqual(2)
  })

  it('previews defining traits on persona cards', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    const preview = screen.getByLabelText('Test Persona defining traits')
    expect(preview).toHaveTextContent('openness')
    expect(preview).toHaveTextContent('80%')
    expect(preview).toHaveTextContent('warmth')
  })

  it('marks the active persona and disables its Activate button', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    const activeCard = screen.getByText('Active Persona').closest('[data-active]')
    expect(activeCard).toHaveAttribute('data-active', 'true')
    const activeButton = activeCard!.querySelector('button')
    expect(activeButton).toBeDisabled()
    // Inactive persona keeps an enabled Activate button
    const inactiveCard = screen.getByText('Test Persona').closest('[data-active]')
    expect(inactiveCard).toHaveAttribute('data-active', 'false')
    expect(inactiveCard!.querySelector('button')).toBeEnabled()
  })

  it('activating a persona calls the controller', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    const inactiveCard = screen.getByText('Test Persona').closest('[data-active]')!
    await act(async () => {
      fireEvent.click(inactiveCard.querySelector('button')!)
    })
    expect(mockActivatePersona).toHaveBeenCalledWith('p1')
  })

  it('displays personality conflicts', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    expect(screen.getByText('Personality Conflicts')).toBeInTheDocument()
    expect(screen.getByText('Conflict')).toBeInTheDocument()
  })

  it('displays preset buttons', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    expect(screen.getByText('Quick Presets')).toBeInTheDocument()
  })

  it('shows personality evolution chart when history has 2+ entries', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    expect(screen.getByText('Personality Evolution')).toBeInTheDocument()
    expect(screen.getAllByTestId('line-chart').length).toBe(2)
  })

  it('saves profile on Save Changes click', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    await act(async () => {
      fireEvent.click(screen.getByText('Save Changes'))
    })
    expect(apiPatch).toHaveBeenCalledWith(
      '/consciousness/personality',
      expect.objectContaining({
        values: expect.any(Array),
        goals: expect.any(Array),
      }),
    )
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Personality saved', 'success')
    })
  })

  it('resets profile on Reset click', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    await act(async () => {
      fireEvent.click(screen.getByText('Reset'))
    })
    expect(consciousnessController.resetPersonality).toHaveBeenCalled()
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Personality reset to defaults', 'success')
    })
  })

  it('does not render the quiz', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    expect(screen.queryByTestId('personality-quiz')).not.toBeInTheDocument()
  })

  it('updates voice slider values', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    const sliders = screen.getAllByRole('slider')
    expect(sliders.length).toBeGreaterThan(0)
    await act(async () => {
      fireEvent.change(sliders[0], { target: { value: '0.9' } })
    })
  })

  it('opens save persona dialog', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    await act(async () => {
      fireEvent.click(screen.getByText('personality.saveCurrent'))
    })
    const saveBtns = screen.getAllByText('personality.savePersona')
    expect(saveBtns.length).toBe(2)
  })

  it('saves persona with name', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    await act(async () => {
      fireEvent.click(screen.getByText('personality.saveCurrent'))
    })
    const input = screen.getByPlaceholderText('My Persona')
    await act(async () => {
      fireEvent.change(input, { target: { value: 'My Test Persona' } })
    })
    await act(async () => {
      const btns = screen.getAllByText('personality.savePersona')
      const saveBtn = btns.find((el) => el.tagName === 'BUTTON')!
      fireEvent.click(saveBtn)
    })
    expect(consciousnessController.savePersona).toHaveBeenCalledWith(
      expect.objectContaining({
        persona_id: 'my-test-persona',
        name: 'My Test Persona',
      }),
    )
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Persona saved', 'success')
    })
  })

  it('activates persona', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    await act(async () => {
      fireEvent.click(screen.getByText('personality.activate'))
    })
    expect(consciousnessController.activatePersona).toHaveBeenCalledWith('p1')
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Persona activated', 'success')
    })
  })

  it('deletes persona after confirmation', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<PersonalityPage />)
    await waitForLoad()
    await act(async () => {
      fireEvent.click(screen.getAllByText('personality.delete')[0])
    })
    expect(consciousnessController.deletePersona).toHaveBeenCalledWith('p1')
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Persona deleted', 'success')
    })
  })

  it('cancels delete when confirmation denied', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false)
    render(<PersonalityPage />)
    await waitForLoad()
    await act(async () => {
      fireEvent.click(screen.getAllByText('personality.delete')[0])
    })
    expect(consciousnessController.deletePersona).not.toHaveBeenCalled()
  })

  it('edits values input field', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    const valuesInput = screen.getByPlaceholderText('helpfulness, honesty, curiosity')
    await act(async () => {
      fireEvent.change(valuesInput, { target: { value: 'courage, wisdom' } })
    })
    expect((valuesInput as HTMLInputElement).value).toBe('courage, wisdom')
  })

  it('cancels persona dialog', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    await act(async () => {
      fireEvent.click(screen.getByText('personality.saveCurrent'))
    })
    await act(async () => {
      fireEvent.click(screen.getByText('Cancel'))
    })
    expect(screen.queryByPlaceholderText('My Persona')).not.toBeInTheDocument()
  })

  it('shows persona card with values', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    expect(screen.getByText('honesty')).toBeInTheDocument()
  })
})
