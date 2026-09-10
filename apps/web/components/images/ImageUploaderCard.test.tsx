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
