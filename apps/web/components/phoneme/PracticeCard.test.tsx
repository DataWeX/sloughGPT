import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import PracticeCard from './PracticeCard'

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: {
    encode: vi.fn().mockResolvedValue({
      text: 'hello', language: 'en', phonemes: ['HH', 'EH', 'L', 'OW'],
    }),
    score: vi.fn(),
  },
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
    { value: 'de', label: 'German' },
    { value: 'fr', label: 'French' },
    { value: 'es', label: 'Spanish' },
    { value: 'it', label: 'Italian' },
    { value: 'pt', label: 'Portuguese' },
  ],
}))

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: vi.fn((selector: any) => selector({
    addToHistory: vi.fn(),
    randomWordTrigger: 0,
    pronunciationStreak: 0,
  })),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: vi.fn((selector: any) => selector({ addToast: vi.fn() })),
}))

vi.mock('./PhonemeSkeleton', () => ({
  default: ({ variant }: any) => <div data-testid="phoneme-skeleton" data-variant={variant} />,
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, className }: any) => <div className={className}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
  CardContent: ({ children, className }: any) => <div className={className}>{children}</div>,
  Input: ({ value, onChange, placeholder, onKeyDown, className, ...props }: any) => (
    <input value={value} onChange={onChange} placeholder={placeholder} onKeyDown={onKeyDown} className={className} {...props} />
  ),
  Button: ({ children, onClick, variant, disabled, className, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} className={className} data-variant={variant} {...props}>{children}</button>
  ),
  Badge: ({ children, variant, className }: any) => <span className={className} data-variant={variant}>{children}</span>,
  Select: ({ children, value, onValueChange }: any) => <select value={value} onChange={e => onValueChange(e.target.value)}>{children}</select>,
  SelectTrigger: ({ children }: any) => <>{children}</>,
  SelectValue: () => null,
  SelectContent: ({ children }: any) => <>{children}</>,
  SelectItem: ({ children, value }: any) => <option value={value}>{children}</option>,
  Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
  IconRefresh: () => <span>refresh</span>,
  IconMic: () => <span>mic</span>,
  IconMicFilled: () => <span>mic-filled</span>,
  IconStop: () => <span>stop</span>,
  IconBolt: () => <span>bolt</span>,
}))

describe('PracticeCard', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('renders without crashing', () => {
    render(<PracticeCard />)
    expect(screen.getByText('Practice Mode')).toBeInTheDocument()
  })

  it('shows target word input and attempt input', () => {
    render(<PracticeCard />)
    expect(screen.getByPlaceholderText('Enter word to practice...')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Type how you would say it...')).toBeInTheDocument()
  })

  it('shows Random Word button', () => {
    render(<PracticeCard />)
    expect(screen.getByText('Random Word')).toBeInTheDocument()
  })

  it('shows Record button', () => {
    render(<PracticeCard />)
    expect(screen.getByText('Record')).toBeInTheDocument()
  })

  it('Score button is disabled when inputs are empty', () => {
    render(<PracticeCard />)
    const scoreBtn = screen.getByText('Score')
    expect(scoreBtn).toBeDisabled()
  })
})
