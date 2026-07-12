// @vitest-environment jsdom

import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: {
    searchFiles: vi.fn(),
    checkDuplicate: vi.fn(),
    gaps: vi.fn(),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (s: any) => s({ addToast: vi.fn() }),
}))

import { KnowledgeOperationsCard } from '../KnowledgeOperationsCard'

describe('KnowledgeOperationsCard', () => {
  it('renders semantic tools card', () => {
    render(<KnowledgeOperationsCard />)
    expect(screen.getByText('Semantic Tools')).toBeDefined()
    expect(screen.getByText('Search codebase')).toBeDefined()
    expect(screen.getByText('Check for duplicates')).toBeDefined()
    expect(screen.getByText('Knowledge gaps')).toBeDefined()
  })
})
