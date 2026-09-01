import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children, className }: any) => <div className={className}>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.filter(Boolean).join(' ')),
    Card: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardContent: passthrough,
    Button: ({ children, onClick, disabled, className, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} disabled={disabled} className={className} aria-label={ariaLabel}>{children}</button>
    ),
    Badge: ({ children, variant, style }: any) => <span data-variant={variant} style={style}>{children}</span>,
    Input: ({ placeholder, disabled, className, 'aria-label': ariaLabel, ...rest }: any) => (
      <input placeholder={placeholder} disabled={disabled} className={className} aria-label={ariaLabel} {...rest} />
    ),
    Separator: () => <hr data-testid="separator" />,
    Switch: ({ checked, onCheckedChange, 'aria-label': ariaLabel }: any) => (
      <button role="switch" aria-checked={checked} aria-label={ariaLabel} onClick={() => onCheckedChange(!checked)} />
    ),
    Checkbox: ({ checked, onCheckedChange, 'aria-label': ariaLabel }: any) => (
      <input type="checkbox" checked={checked} aria-label={ariaLabel} onChange={(e) => onCheckedChange(e.target.checked)} />
    ),
    Progress: ({ value, className, 'aria-label': ariaLabel }: any) => (
      <div role="progressbar" aria-valuenow={value} className={className} aria-label={ariaLabel} />
    ),
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ title, subtitle, children }: any) => (
    <div data-testid="page-container">
      <h1>{title}</h1>
      {subtitle && <p>{subtitle}</p>}
      {children}
    </div>
  ),
}))

import MagazinePage from './page'

describe('MagazinePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders without crashing', () => {
    render(<MagazinePage />)
    expect(screen.getByTestId('page-container')).toBeTruthy()
  })

  it('renders page header with title and subtitle', () => {
    render(<MagazinePage />)
    expect(screen.getAllByRole('heading', { name: /Noir Violet/ }).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('A design system reference for sloughGPT').length).toBeGreaterThanOrEqual(1)
  })

  it('renders hero section with primary color swatch', () => {
    render(<MagazinePage />)
    expect(screen.getAllByRole('img', { name: /Primary violet color swatch/ }).length).toBe(1)
  })

  it('renders hero badges', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('v1.0').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Design System').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Accent: Terracotta').length).toBe(1)
  })

  it('renders hero tagline', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText(/Warm, sophisticated, technical/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/A calm confidence/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders section headers', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Color Palette').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Typography Scale').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Component Gallery').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Spacing & Radius').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Interactive States').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Dark Mode Preview').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Accent Themes').length).toBeGreaterThanOrEqual(1)
  })

  it('renders light mode color palette card', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Light Mode').length).toBeGreaterThanOrEqual(1)
  })

  it('renders dark mode color palette card', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Dark Mode').length).toBeGreaterThanOrEqual(1)
  })

  it('renders typography scale entries', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Page Title').length).toBe(1)
    expect(screen.getAllByText('Section Title').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Body').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Caption').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Mono').length).toBeGreaterThanOrEqual(1)
  })

  it('renders font families', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Rubik').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Lato').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('JetBrains Mono').length).toBeGreaterThanOrEqual(1)
  })

  it('renders button variants', () => {
    render(<MagazinePage />)
    const buttons = screen.getAllByRole('button')
    const buttonTexts = buttons.map(b => b.textContent)
    expect(buttonTexts).toContain('Primary')
    expect(buttonTexts).toContain('Secondary')
    expect(buttonTexts).toContain('Destructive')
    expect(buttonTexts).toContain('Outline')
    expect(buttonTexts).toContain('Ghost')
    expect(buttonTexts).toContain('Link')
  })

  it('renders disabled buttons', () => {
    render(<MagazinePage />)
    const disabledButtons = screen.getAllByRole('button').filter(b => b.hasAttribute('disabled'))
    const disabledTexts = disabledButtons.map(b => b.textContent)
    expect(disabledTexts.filter(t => t === 'Disabled').length).toBeGreaterThanOrEqual(1)
  })

  it('renders form input elements', () => {
    render(<MagazinePage />)
    expect(screen.getByLabelText('Example input')).toBeTruthy()
    expect(screen.getByLabelText('Disabled input')).toBeTruthy()
    expect(screen.getByLabelText('Disabled input')).toBeDisabled()
  })

  it('renders switch toggle and toggles on click', () => {
    render(<MagazinePage />)
    const switchEl = screen.getByRole('switch', { name: 'Example switch' })
    expect(switchEl).toHaveAttribute('aria-checked', 'false')
    fireEvent.click(switchEl)
    expect(switchEl).toHaveAttribute('aria-checked', 'true')
  })

  it('renders checkbox and toggles on click', () => {
    render(<MagazinePage />)
    const checkbox = screen.getByRole('checkbox', { name: 'Example checkbox' })
    expect(checkbox).not.toBeChecked()
    fireEvent.click(checkbox)
    expect(checkbox).toBeChecked()
  })

  it('renders progress bar with correct value', () => {
    render(<MagazinePage />)
    const progress = screen.getByRole('progressbar')
    expect(progress).toHaveAttribute('aria-valuenow', '64')
  })

  it('renders spacing scale tokens', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('gap-1').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('gap-2').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('gap-4').length).toBeGreaterThanOrEqual(1)
  })

  it('renders border radius scale', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('rounded-none').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('rounded-md').length).toBeGreaterThanOrEqual(1)
  })

  it('renders button states section', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Hover').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Focus').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Active').length).toBeGreaterThanOrEqual(1)
  })

  it('renders hover preview with hoverable card', () => {
    render(<MagazinePage />)
    const hoverCard = screen.getByRole('button', { name: 'Hoverable card demo' })
    expect(hoverCard).toBeTruthy()
  })

  it('hoverable card responds to mouse events', () => {
    render(<MagazinePage />)
    const hoverCard = screen.getByRole('button', { name: 'Hoverable card demo' })
    fireEvent.mouseEnter(hoverCard)
    fireEvent.mouseLeave(hoverCard)
  })

  it('renders dark mode preview section', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Dark Mode Palette').length).toBeGreaterThanOrEqual(1)
  })

  it('renders dark mode sample buttons and badges', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Primary Button').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Outline Button').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Success Badge').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Accent Badge').length).toBeGreaterThanOrEqual(1)
  })

  it('renders all seven accent themes', () => {
    render(<MagazinePage />)
    const accentNames = ['Blue', 'Purple', 'Pink', 'Red', 'Orange', 'Green', 'Teal']
    for (const name of accentNames) {
      expect(screen.getAllByText(name).length).toBeGreaterThanOrEqual(1)
    }
  })

  it('renders Apply buttons for accent themes', () => {
    render(<MagazinePage />)
    const allButtons = screen.getAllByRole('button')
    const applyButtons = allButtons.filter(b => b.textContent === 'Apply')
    expect(applyButtons.length).toBe(7)
  })

  it('renders focus ring demo input', () => {
    render(<MagazinePage />)
    const inputs = screen.getAllByPlaceholderText('Tab here to see focus ring...')
    expect(inputs.length).toBeGreaterThanOrEqual(1)
  })

  it('renders footer with branding text', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText(/sloughGPT.*Noir Violet Design System/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Locked').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('v1.0').length).toBeGreaterThanOrEqual(1)
  })

  it('renders separators between major sections', () => {
    const { container } = render(<MagazinePage />)
    const separators = container.querySelectorAll('[data-testid="separator"]')
    expect(separators.length).toBeGreaterThanOrEqual(5)
  })

  it('renders section numbers', () => {
    render(<MagazinePage />)
    const sectionNums = ['02', '03', '04', '05', '06', '07', '08']
    for (const num of sectionNums) {
      expect(screen.getAllByText(num).length).toBeGreaterThanOrEqual(1)
    }
  })

  it('renders print stylesheet', () => {
    const { container } = render(<MagazinePage />)
    const styleTags = container.querySelectorAll('style')
    expect(styleTags.length).toBeGreaterThanOrEqual(1)
    expect(styleTags[0].textContent).toContain('@media print')
  })

  it('renders shadow depth scale', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Shadow Depth').length).toBeGreaterThanOrEqual(1)
  })

  it('renders font families card', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Font Families').length).toBeGreaterThanOrEqual(1)
  })

  it('renders form elements card', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Form Elements').length).toBeGreaterThanOrEqual(1)
  })

  it('renders cards shadow card', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Cards').length).toBeGreaterThanOrEqual(1)
  })

  it('renders badges card', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Badges').length).toBeGreaterThanOrEqual(1)
  })

  it('renders theme in context card', () => {
    render(<MagazinePage />)
    expect(screen.getAllByText('Theme in Context').length).toBeGreaterThanOrEqual(1)
  })
})
