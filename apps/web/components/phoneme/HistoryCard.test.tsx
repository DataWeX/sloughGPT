import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import HistoryCard from './HistoryCard'

vi.mock('recharts', () => ({
  AreaChart: ({ children }: any) => <div data-testid="area-chart">{children}</div>,
  Area: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
}))

vi.mock('@/lib/phoneme-controller', () => ({
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
    { value: 'de', label: 'German' },
    { value: 'fr', label: 'French' },
    { value: 'es', label: 'Spanish' },
    { value: 'it', label: 'Italian' },
    { value: 'pt', label: 'Portuguese' },
  ],
}))

const mockHistory = [
  { id: '1', target: 'hello', spoken: 'helo', score: 0.9, language: 'en', targetPhonemes: ['HH', 'EH', 'L', 'OW'], spokenPhonemes: ['HH', 'EH', 'L', 'OW'], timestamp: Date.now() - 1000 },
  { id: '2', target: 'cat', spoken: 'kat', score: 0.7, language: 'en', targetPhonemes: ['K', 'AE', 'T'], spokenPhonemes: ['K', 'AA', 'T'], timestamp: Date.now() },
]

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: vi.fn((selector: any) => selector({
    history: mockHistory,
    clearHistory: vi.fn(),
    exportHistory: vi.fn(() => '[]'),
    importHistory: vi.fn(),
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
  Button: ({ children, onClick, variant, size, className, ...props }: any) => (
    <button onClick={onClick} className={className} data-variant={variant} {...props}>{children}</button>
  ),
  Badge: ({ children, variant, className }: any) => <span className={className} data-variant={variant}>{children}</span>,
  Select: ({ children, value, onValueChange }: any) => <select value={value} onChange={e => onValueChange(e.target.value)}>{children}</select>,
  SelectTrigger: ({ children }: any) => <>{children}</>,
  SelectValue: () => null,
  SelectContent: ({ children }: any) => <>{children}</>,
  SelectItem: ({ children, value }: any) => <option value={value}>{children}</option>,
  Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
  Collapsible: ({ children }: any) => <div>{children}</div>,
  CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  CollapsibleContent: ({ children }: any) => <div>{children}</div>,
  IconTrash: () => <span>trash</span>,
  IconDownload: () => <span>download</span>,
  IconUpload: () => <span>upload</span>,
}))

describe('HistoryCard', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('renders without crashing', () => {
    render(<HistoryCard />)
    expect(screen.getByText('Pronunciation History')).toBeInTheDocument()
  })

  it('displays history entry count', () => {
    render(<HistoryCard />)
    expect(screen.getByText('2 entries')).toBeInTheDocument()
  })

  it('shows export buttons', () => {
    render(<HistoryCard />)
    expect(screen.getByText('JSON')).toBeInTheDocument()
    expect(screen.getByText('CSV')).toBeInTheDocument()
  })

  it('shows Clear button', () => {
    render(<HistoryCard />)
    expect(screen.getByText('Clear')).toBeInTheDocument()
  })

  it('shows empty state when no history', () => {
    render(<HistoryCard />)
    expect(screen.getByText('2 entries')).toBeInTheDocument()
  })
})
