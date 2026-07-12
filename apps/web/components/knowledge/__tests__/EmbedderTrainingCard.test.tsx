// @vitest-environment jsdom

import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: {
    getEmbedderStatus: vi.fn().mockResolvedValue({ trained: false, info: null }),
    trainEmbedder: vi.fn(),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (s: any) => s({ addToast: vi.fn() }),
}))

import { EmbedderTrainingCard } from '../EmbedderTrainingCard'

describe('EmbedderTrainingCard', () => {
  it('renders untrained state with train button', async () => {
    render(<EmbedderTrainingCard />)
    expect(screen.getByText('Text Embedder')).toBeDefined()
    expect(screen.getByText('Train Embedder')).toBeDefined()
    await waitFor(() => {
      expect(screen.getByText(/Train a SloNet text embedder/)).toBeDefined()
    })
  })
})
