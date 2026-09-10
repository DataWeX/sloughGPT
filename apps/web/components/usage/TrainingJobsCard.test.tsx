/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { TrainingJobsCard } from './TrainingJobsCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

describe('TrainingJobsCard', () => {
  it('renders the card title', () => {
    render(<TrainingJobsCard completed={0} running={0} queued={0} failed={0} totalMinutes={0} />)
    expect(screen.getByText('Training Jobs')).toBeDefined()
  })

  it('renders completed count', () => {
    render(<TrainingJobsCard completed={12} running={0} queued={0} failed={0} totalMinutes={0} />)
    expect(screen.getByText('12')).toBeDefined()
    expect(screen.getByText('Completed')).toBeDefined()
  })

  it('renders running count', () => {
    render(<TrainingJobsCard completed={0} running={3} queued={0} failed={0} totalMinutes={0} />)
    expect(screen.getByText('3')).toBeDefined()
    expect(screen.getByText('Running')).toBeDefined()
  })

  it('renders total training time', () => {
    render(<TrainingJobsCard completed={5} running={1} queued={2} failed={0} totalMinutes={120} />)
    expect(screen.getByText('Total training time: 120 minutes')).toBeDefined()
  })

  it('renders failed count', () => {
    render(<TrainingJobsCard completed={0} running={0} queued={0} failed={7} totalMinutes={0} />)
    expect(screen.getByText('7')).toBeDefined()
    expect(screen.getByText('Failed')).toBeDefined()
  })
})
