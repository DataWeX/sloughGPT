import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Panel } from './Panel'

describe('Panel', () => {
  it('renders title and children', () => {
    render(<Panel title="My Panel"><p>Content</p></Panel>)
    expect(screen.getByText('My Panel')).toBeInTheDocument()
    expect(screen.getByText('Content')).toBeInTheDocument()
  })

  it('renders actions when provided', () => {
    render(
      <Panel title="With Actions" actions={<button>Action</button>}>
        Body
      </Panel>
    )
    expect(screen.getByRole('button', { name: 'Action' })).toBeInTheDocument()
  })

  it('sets data-testid when provided', () => {
    render(<Panel title="T" testId="panel-1">X</Panel>)
    expect(screen.getByTestId('panel-1')).toBeInTheDocument()
  })
})
