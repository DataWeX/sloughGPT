// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/images-controller', () => ({
  imagesController: { gallery: vi.fn(), styles: vi.fn() },
}))
vi.mock('@/lib/http-client', () => ({
  apiPost: vi.fn(),
}))
vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))
vi.mock('@/components/images/ImageGalleryInsightsCard', () => ({
  ImageGalleryInsightsCard: () => <div data-testid="insights" />,
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (s: any) => s({ addToast: vi.fn() }),
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  ActionCard: ({ title, actions, children, ...p }: any) => <div data-testid="action-card" {...p}>{title}{actions}{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Textarea: ({ value, onChange, ...p }: any) => <textarea value={value} onChange={e => onChange(e.target.value)} {...p} />,
  IconRefresh: () => <span>↻</span>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
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
}))

import { ImageSection } from './ImageSection'
import { imagesController } from '@/lib/images-controller'

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(imagesController.gallery).mockResolvedValue([] as any)
  vi.mocked(imagesController.styles).mockResolvedValue([] as any)
})

afterEach(() => cleanup())

describe('ImageSection', () => {
  it('calls gallery and styles on mount', async () => {
    render(<ImageSection />)
    await waitFor(() => {
      expect(imagesController.gallery).toHaveBeenCalled()
      expect(imagesController.styles).toHaveBeenCalled()
    })
  })

  it('renders without crashing', async () => {
    render(<ImageSection />)
    await waitFor(() => {
      expect(imagesController.gallery).toHaveBeenCalled()
    })
  })
})
