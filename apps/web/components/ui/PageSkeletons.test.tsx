import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import {
  KnowledgeStatsSkeleton,
  KnowledgeCategoryChartSkeleton,
  KnowledgeTopicsSkeleton,
  KnowledgeAdapterSkeleton,
  KnowledgeRAGSkeleton,
  KnowledgeItemsSkeleton,
  KnowledgePageSkeleton,
  DatasetListSkeleton,
  DatasetsPageSkeleton,
  ModelStatusCardSkeleton,
  PersonalitiesCardSkeleton,
  ModelCatalogSkeleton,
  ModelsPageSkeleton,
} from './PageSkeletons'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children, className }: any) => <div className={className}>{children}</div>
  return {
    Skeleton: ({ className }: { className?: string }) => <div data-testid="skeleton" className={className} />,
    Card: passthrough,
    CardContent: passthrough,
  }
})

describe('PageSkeletons', () => {
  describe('KnowledgeStatsSkeleton', () => {
    it('renders 4 stat card skeletons', () => {
      render(<KnowledgeStatsSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(4)
    })
  })

  describe('KnowledgeCategoryChartSkeleton', () => {
    it('renders chart skeleton with bar placeholders', () => {
      const { container } = render(<KnowledgeCategoryChartSkeleton />)
      expect(container.querySelector('.h-24')).toBeTruthy()
    })
  })

  describe('KnowledgeTopicsSkeleton', () => {
    it('renders topic bar skeletons', () => {
      render(<KnowledgeTopicsSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(5)
    })
  })

  describe('KnowledgeAdapterSkeleton', () => {
    it('renders adapter card skeleton', () => {
      render(<KnowledgeAdapterSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(3)
    })
  })

  describe('KnowledgeRAGSkeleton', () => {
    it('renders RAG card skeleton', () => {
      render(<KnowledgeRAGSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(4)
    })
  })

  describe('KnowledgeItemsSkeleton', () => {
    it('renders default 5 item skeletons', () => {
      render(<KnowledgeItemsSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(15)
    })
    it('renders custom count', () => {
      render(<KnowledgeItemsSkeleton count={2} />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(6)
    })
  })

  describe('KnowledgePageSkeleton', () => {
    it('renders all knowledge skeleton sections', () => {
      render(<KnowledgePageSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(20)
    })
  })

  describe('DatasetListSkeleton', () => {
    it('renders default 4 dataset skeletons', () => {
      render(<DatasetListSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(20)
    })
    it('renders custom count', () => {
      render(<DatasetListSkeleton count={2} />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(10)
    })
  })

  describe('DatasetsPageSkeleton', () => {
    it('renders toolbar and list skeletons', () => {
      render(<DatasetsPageSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(20)
    })
  })

  describe('ModelStatusCardSkeleton', () => {
    it('renders status card skeleton', () => {
      render(<ModelStatusCardSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(5)
    })
  })

  describe('PersonalitiesCardSkeleton', () => {
    it('renders personality grid skeletons', () => {
      render(<PersonalitiesCardSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(18)
    })
  })

  describe('ModelCatalogSkeleton', () => {
    it('renders catalog list skeletons', () => {
      render(<ModelCatalogSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(12)
    })
  })

  describe('ModelsPageSkeleton', () => {
    it('renders all model skeleton sections', () => {
      render(<ModelsPageSkeleton />)
      expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(30)
    })
  })
})
