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
  
    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
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
