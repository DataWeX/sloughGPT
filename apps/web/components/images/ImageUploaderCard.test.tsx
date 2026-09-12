// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,

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

import { ImageUploaderCard } from './ImageUploaderCard'

afterEach(() => { cleanup(); localStorage.clear() })

describe('ImageUploaderCard', () => {
  it('renders upload area', () => {
    render(<ImageUploaderCard />)
    expect(screen.getByText('Image Upload')).toBeTruthy()
    expect(screen.getByText(/Drag.*drop/)).toBeTruthy()
    expect(screen.getByText('browse')).toBeTruthy()
  })

  it('loads existing uploads from localStorage', () => {
    localStorage.setItem('sloughgpt-image-uploads', JSON.stringify([
      { id: 'img-1', name: 'photo.png', size: 1024, type: 'image/png', dataUrl: 'data:image/png;base64,abc', timestamp: 1 },
      { id: 'img-2', name: 'shot.jpg', size: 2048, type: 'image/jpeg', dataUrl: 'data:image/jpeg;base64,def', timestamp: 2 },
    ]))
    render(<ImageUploaderCard />)
    expect(screen.getByText('2 uploads')).toBeTruthy()
    expect(screen.getByText('photo.png')).toBeTruthy()
    expect(screen.getByText('shot.jpg')).toBeTruthy()
  })

  it('deletes an upload', async () => {
    localStorage.setItem('sloughgpt-image-uploads', JSON.stringify([
      { id: 'img-1', name: 'del.png', size: 500, type: 'image/png', dataUrl: 'data:image/png;base64,x', timestamp: 1 },
    ]))
    render(<ImageUploaderCard />)
    expect(screen.getByText('del.png')).toBeTruthy()
    const delBtn = screen.getAllByRole('button').find(b => b.textContent === '✕')
    if (delBtn) fireEvent.click(delBtn)
    await waitFor(() => {
      expect(screen.queryByText('del.png')).toBeNull()
    })
  })

  it('opens file picker on button click', () => {
    render(<ImageUploaderCard />)
    const uploadBtn = screen.getAllByRole('button').find(b => b.textContent === '+ Upload')
    fireEvent.click(uploadBtn!)
    expect(screen.getByTestId('image-uploader').querySelector('input[type="file"]')).toBeTruthy()
  })

  it('handles drag over', () => {
    render(<ImageUploaderCard />)
    const dropZone = screen.getByText(/Drag.*drop/).closest('div')!.parentElement!
    fireEvent.dragOver(dropZone)
    expect(dropZone.className).toContain('border-primary')
  })
})
