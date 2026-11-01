import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import FlashcardCard from './FlashcardCard'

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: { encode: vi.fn(), score: vi.fn() },
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
    flashcardSRS: {},
    updateFlashcardSRS: vi.fn(),
    resetFlashcardSRS: vi.fn(),
  })),
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
  IconRefresh: () => <span>refresh</span>,
  IconCheck: () => <span>check</span>,
  IconX: () => <span>x</span>,
}))

describe('FlashcardCard', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('renders without crashing', () => {
    render(<FlashcardCard />)
    expect(screen.getByText('Pronunciation Flashcards')).toBeInTheDocument()
  })

  it('displays the first flashcard word', () => {
    render(<FlashcardCard />)
    expect(screen.getAllByText('cat').length).toBeGreaterThanOrEqual(1)
  })

  it('flips card on click', () => {
    const { container } = render(<FlashcardCard />)
    const flipArea = container.querySelector('.cursor-pointer')!
    fireEvent.click(flipArea)
    expect(screen.getByText('K')).toBeInTheDocument()
    expect(screen.getByText('AE')).toBeInTheDocument()
    expect(screen.getByText('T')).toBeInTheDocument()
  })

  it('advances to next card on Know It', () => {
    render(<FlashcardCard />)
    const knowBtn = screen.getAllByRole('button').find(b => b.textContent?.includes('Know It'))!
    fireEvent.click(knowBtn)
    expect(screen.getAllByText('church').length).toBeGreaterThanOrEqual(1)
  })

  it('shows mastered and due badge counts', () => {
    render(<FlashcardCard />)
    expect(screen.getByText('0 mastered')).toBeInTheDocument()
    expect(screen.getByText(/due/)).toBeInTheDocument()
  })
})
