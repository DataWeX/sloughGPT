/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { PersonalityCoreIdentity } from './PersonalityCoreIdentity'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardDescription: ({ children, ...props }: any) => <div data-testid="card-description" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Input: ({ value, onChange, placeholder, ...props }: any) => (
    <input data-testid="input" value={value} onChange={onChange} placeholder={placeholder} {...props} />
  ),
}))

afterEach(() => cleanup())

describe('PersonalityCoreIdentity', () => {
  const defaultProps = {
    values: 'honesty, curiosity',
    goals: 'help users',
    interests: 'AI, science',
    avoid: 'being rude',
    onValuesChange: vi.fn(),
    onGoalsChange: vi.fn(),
    onInterestsChange: vi.fn(),
    onAvoidChange: vi.fn(),
  }

  it('renders title "Core Identity"', () => {
    render(<PersonalityCoreIdentity {...defaultProps} />)
    expect(screen.getByText('Core Identity')).toBeDefined()
  })

  it('renders four input fields', () => {
    render(<PersonalityCoreIdentity {...defaultProps} />)
    const inputs = screen.getAllByTestId('input')
    expect(inputs).toHaveLength(4)
  })

  it('displays current values', () => {
    render(<PersonalityCoreIdentity {...defaultProps} />)
    const inputs = screen.getAllByTestId('input')
    expect((inputs[0] as HTMLInputElement).value).toBe('honesty, curiosity')
    expect((inputs[1] as HTMLInputElement).value).toBe('help users')
  })

  it('renders labels for each field', () => {
    render(<PersonalityCoreIdentity {...defaultProps} />)
    expect(screen.getByText('Values (comma-separated)')).toBeDefined()
    expect(screen.getByText('Goals (comma-separated)')).toBeDefined()
    expect(screen.getByText('Interests (comma-separated)')).toBeDefined()
    expect(screen.getByText('Avoid (comma-separated)')).toBeDefined()
  })

  it('renders placeholders', () => {
    render(<PersonalityCoreIdentity {...defaultProps} />)
    expect(screen.getByPlaceholderText('helpfulness, honesty, curiosity')).toBeDefined()
    expect(screen.getByPlaceholderText('AI, programming, science')).toBeDefined()
  })
})
