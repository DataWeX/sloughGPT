/// <reference types="vitest" />
import '@testing-library/jest-dom/vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, it, expect } from 'vitest'
import { KnowledgeStatsCard } from './KnowledgeStatsCard'

afterEach(() => cleanup())

describe('KnowledgeStatsCard', () => {
  it('renders title', () => {
    render(<KnowledgeStatsCard totalItems={0} topics={[]} avgImportance={0} sources={{}} />)
    expect(screen.getByText('Knowledge Stats')).toBeInTheDocument()
  })

  it('displays total items count', () => {
    render(<KnowledgeStatsCard totalItems={42} topics={['a', 'b']} avgImportance={0.8} sources={{ web: 1 }} />)
    expect(screen.getByText('42')).toBeInTheDocument()
  })

  it('displays topic count', () => {
    render(<KnowledgeStatsCard totalItems={10} topics={['a', 'b', 'c']} avgImportance={0.5} sources={{}} />)
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('displays average importance', () => {
    render(<KnowledgeStatsCard totalItems={5} topics={[]} avgImportance={0.75} sources={{}} />)
    expect(screen.getByText('0.75')).toBeInTheDocument()
  })

  it('displays source count', () => {
    render(<KnowledgeStatsCard totalItems={5} topics={[]} avgImportance={0.5} sources={{ web: 1, file: 2 }} />)
    expect(screen.getByText('2')).toBeInTheDocument()
  })
})
