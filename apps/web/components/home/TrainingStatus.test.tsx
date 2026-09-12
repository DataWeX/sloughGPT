import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { TrainingStatus } from './TrainingStatus'

vi.mock('next/link', () => ({
  default: ({ children, href, ...props }: any) => <a href={href} {...props}>{children}</a>,
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({
    t: (key: string, params?: any) => {
      if (key === 'home.apiOffline.title') return 'API Offline'
      if (key === 'home.apiOffline.body') return `Cannot reach ${params?.url || 'API'}`
      return key
    },
  }),
}))

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children, className }: any) => <div className={className}>{children}</div>
  return {
    Card: passthrough,
    CardContent: passthrough,
    CardHeader: passthrough,
    CardTitle: passthrough,
    cn: (...args: any[]) => args.filter(Boolean).join(' '),
  
Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
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

const baseProps = {
  apiStatus: 'online' as string,
  modelStatus: { loaded: true, model: 'gpt2' },
  modelReadiness: { ready: true, phase: 'ready', step: 9, total: 9, message: 'Ready' },
  runningTraining: null as { name: string; status_message: string } | null,
}

describe('TrainingStatus', () => {
  describe('offline state', () => {
    it('shows startup progress when initializing', () => {
      const { container } = render(<TrainingStatus {...baseProps} apiStatus="offline" modelReadiness={{ ready: false, phase: 'loading', step: 5, total: 9, message: 'Loading weights...' }} />)
      expect(container.textContent).toMatch(/Starting up/)
      expect(container.textContent).toContain('Loading weights...')
    })

    it('shows offline card when phase is unknown', () => {
      const { container } = render(<TrainingStatus {...baseProps} apiStatus="offline" modelReadiness={{ ready: false, phase: 'unknown', step: 0, total: 9, message: '' }} />)
      expect(container.textContent).toContain('API Offline')
    })
  })

  describe('running training', () => {
    it('shows training banner when training is running', () => {
      const { container } = render(<TrainingStatus {...baseProps} runningTraining={{ name: 'my-run', status_message: 'Epoch 3/10' }} />)
      expect(container.textContent).toMatch(/Training: my-run/)
      expect(container.textContent).toContain('Epoch 3/10')
    })

    it('links to training page', () => {
      const { container } = render(<TrainingStatus {...baseProps} runningTraining={{ name: 'run-1', status_message: 'done' }} />)
      const links = container.querySelectorAll('a')
      const trainingLink = Array.from(links).find(a => a.getAttribute('href') === '/training')
      expect(trainingLink).toBeDefined()
      expect(trainingLink!.textContent).toMatch(/Training: run-1/)
    })

    it('hides banner when no training running', () => {
      const { container } = render(<TrainingStatus {...baseProps} />)
      expect(container.textContent).not.toMatch(/Training:/)
    })
  })

  describe('model loading', () => {
    it('shows loading progress when model not ready', () => {
      const { container } = render(<TrainingStatus {...baseProps} modelReadiness={{ ready: false, phase: 'loading', step: 3, total: 9, message: 'Loading...' }} />)
      expect(container.textContent).toContain('Model loading')
      expect(container.textContent).toContain('Loading...')
    })
  })

  describe('no model loaded', () => {
    it('shows no model card with link to models', () => {
      const { container } = render(<TrainingStatus {...baseProps} modelStatus={{ loaded: false, model: null }} />)
      expect(container.textContent).toContain('No model loaded')
      const links = container.querySelectorAll('a')
      const modelsLink = Array.from(links).find(a => a.getAttribute('href') === '/models')
      expect(modelsLink).toBeDefined()
      expect(modelsLink!.textContent).toContain('Open Models')
    })
  })
})
