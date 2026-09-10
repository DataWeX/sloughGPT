import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act, waitFor, fireEvent } from '@testing-library/react'
import PersonalityPage from './page'

vi.mock('@/lib/config', () => ({ PUBLIC_API_URL: 'http://localhost:8000' }))

const mockAddToast = vi.fn()
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: typeof mockAddToast }) => typeof mockAddToast) =>
    selector({ addToast: mockAddToast }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (key: string) => key, locale: 'en' }),
}))

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title }: { children: React.ReactNode; title: string }) => (
    <div data-testid="page-container" data-title={title}>{children}</div>
  ),
}))

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}))

vi.mock('./PersonalityQuiz', () => ({
  PersonalityQuiz: ({ onApply }: { onApply: (preset: string) => void }) => (
    <div data-testid="personality-quiz">
      <button onClick={() => onApply('creative')}>Apply Creative</button>
    </div>
  ),
}))

const mockProfile = {
  values: ['honesty', 'curiosity'],
  goals: ['be helpful'],
  voice: { formality: 0.5, warmth: 0.7, confidence: 0.6, humor: 0.3, verbosity: 0.4, empathy: 0.8 },
  style: { use_examples: true, ask_follow_ups: false, acknowledge_uncertainty: true, use_analogies: false, break_down_complex_topics: true },
  traits: { openness: 0.6, conscientiousness: 0.7, extraversion: 0.4, agreeableness: 0.8, neuroticism: 0.3 },
  interests: ['AI', 'science'],
  avoid: ['rudeness'],
}

const mockPresets = { data: { names: ['default', 'formal', 'creative'] } }
const mockHistory = {
  data: {
    history: [
      { timestamp: Date.now() / 1000 - 3600, voice: { warmth: 0.5 }, traits: { openness: 0.4 } },
      { timestamp: Date.now() / 1000, voice: { warmth: 0.7 }, traits: { openness: 0.6 } },
    ],
  },
}
const mockConflicts = { data: { conflicts: [{ type: 'style', severity: 'medium', message: 'Conflict', fields: ['humor', 'formality'] }] } }
const mockPersonas = { personas: [{ id: 'p1', name: 'Test Persona', values: ['honesty'] }] }

let fetchMock: ReturnType<typeof vi.fn>

beforeEach(() => {
  fetchMock = vi.fn().mockImplementation((url: string, opts?: any) => {
    if (typeof url === 'string') {
      if (url.includes('/personality') && !url.includes('preset') && !url.includes('history') && !url.includes('conflict') && !url.includes('reset') && !url.includes('save')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ data: mockProfile }) })
      }
      if (url.includes('/presets') && !url.includes('apply')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(mockPresets) })
      }
      if (url.includes('/history')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(mockHistory) })
      }
      if (url.includes('/conflicts')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(mockConflicts) })
      }
      if (url.includes('/personas') && !url.includes('save') && !url.includes('activate')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(mockPersonas) })
      }
      if (url.includes('/reset')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ data: mockProfile }) })
      }
      if (url.includes('/presets/apply')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ data: mockProfile }) })
      }
      if (url.includes('/personas/save')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
      }
      if (url.includes('/activate')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
      }
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
  })
  global.fetch = fetchMock
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
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/consciousness/personality',
      expect.objectContaining({ method: 'PATCH' })
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
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/consciousness/personality/reset',
      expect.objectContaining({ method: 'POST' })
    )
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Personality reset to defaults', 'success')
    })
  })

  it('applies preset via quiz', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    await act(async () => {
      fireEvent.click(screen.getByText('Apply Creative'))
    })
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/consciousness/personality/presets/apply',
      expect.objectContaining({ method: 'POST' })
    )
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Applied "creative" preset', 'success')
    })
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
      const saveBtn = btns.find(el => el.tagName === 'BUTTON')!
      fireEvent.click(saveBtn)
    })
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/consciousness/personas/save',
      expect.objectContaining({ method: 'POST' })
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
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/consciousness/personas/p1/activate',
      expect.objectContaining({ method: 'POST' })
    )
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Persona activated', 'success')
    })
  })

  it('deletes persona after confirmation', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<PersonalityPage />)
    await waitForLoad()
    await act(async () => {
      fireEvent.click(screen.getByText('personality.delete'))
    })
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/consciousness/personas/p1',
      expect.objectContaining({ method: 'DELETE' })
    )
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Persona deleted', 'success')
    })
  })

  it('cancels delete when confirmation denied', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false)
    render(<PersonalityPage />)
    await waitForLoad()
    await act(async () => {
      fireEvent.click(screen.getByText('personality.delete'))
    })
    const deleteCalls = fetchMock.mock.calls.filter((c: any) => c[1]?.method === 'DELETE')
    expect(deleteCalls.length).toBe(0)
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

  it('renders personality quiz component', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    expect(screen.getByTestId('personality-quiz')).toBeInTheDocument()
  })

  it('shows persona card with values', async () => {
    render(<PersonalityPage />)
    await waitForLoad()
    expect(screen.getByText('honesty')).toBeInTheDocument()
  })
})
