// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Input: (props: any) => <input {...props} />,
  Switch: ({ checked, onCheckedChange, ...props }: any) => (
    <input type="checkbox" checked={checked} onChange={e => onCheckedChange(e.target.checked)} {...props} />
  ),

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
}))

import { SettingsTrainingCard } from './SettingsTrainingCard'

afterEach(() => cleanup())

describe('SettingsTrainingCard', () => {
  it('renders title', () => {
    render(<SettingsTrainingCard
      preferredModel="" preferredMethod="" maxCheckpoints={10} autoTrainThreshold={0.8}
      autoTrain={false} enableTracking={true}
      onPreferredModelChange={() => {}} onPreferredMethodChange={() => {}}
      onMaxCheckpointsChange={() => {}} onAutoTrainThresholdChange={() => {}}
      onAutoTrainChange={() => {}} onEnableTrackingChange={() => {}}
    />)
    expect(screen.getByText('Training')).toBeTruthy()
  })

  it('renders preferred model input', () => {
    render(<SettingsTrainingCard
      preferredModel="gpt2" preferredMethod="" maxCheckpoints={10} autoTrainThreshold={0.8}
      autoTrain={false} enableTracking={true}
      onPreferredModelChange={() => {}} onPreferredMethodChange={() => {}}
      onMaxCheckpointsChange={() => {}} onAutoTrainThresholdChange={() => {}}
      onAutoTrainChange={() => {}} onEnableTrackingChange={() => {}}
    />)
    expect(screen.getByDisplayValue('gpt2')).toBeTruthy()
  })

  it('renders auto-train toggle', () => {
    render(<SettingsTrainingCard
      preferredModel="" preferredMethod="" maxCheckpoints={10} autoTrainThreshold={0.8}
      autoTrain={true} enableTracking={true}
      onPreferredModelChange={() => {}} onPreferredMethodChange={() => {}}
      onMaxCheckpointsChange={() => {}} onAutoTrainThresholdChange={() => {}}
      onAutoTrainChange={() => {}} onEnableTrackingChange={() => {}}
    />)
    expect(screen.getByText('Auto-train')).toBeTruthy()
  })
})
