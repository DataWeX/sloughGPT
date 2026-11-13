'use client'

import { render, screen, cleanup } from '@testing-library/react'
import { describe, it, expect, vi, afterEach } from 'vitest'

import { PageContainer } from './PageContainer'

afterEach(cleanup)

describe('PageContainer', () => {
  it('renders title and children', () => {
    render(
      <PageContainer title="Test Page">
        <p>Content here</p>
      </PageContainer>,
    )
    expect(screen.getByText('Test Page')).toBeInTheDocument()
    expect(screen.getByText('Content here')).toBeInTheDocument()
  })

  it('renders subtitle when provided', () => {
    render(
      <PageContainer title="Settings" subtitle="Configure your app">
        <p>Content</p>
      </PageContainer>,
    )
    expect(screen.getByText('Configure your app')).toBeInTheDocument()
  })

  it('renders headerRight when provided', () => {
    render(
      <PageContainer title="Models" headerRight={<button>Action</button>}>
        <p>Content</p>
      </PageContainer>,
    )
    expect(screen.getByText('Action')).toBeInTheDocument()
  })

  it('renders heading as h1', () => {
    render(
      <PageContainer title="Dashboard">
        <p>Content</p>
      </PageContainer>,
    )
    expect(screen.getByRole('heading', { level: 1, name: 'Dashboard' })).toBeInTheDocument()
  })

  // ── Loading ──────────────────────────────────────────

  it('shows default loading skeleton when loading=true', () => {
    const { container } = render(
      <PageContainer title="Settings" loading>
        <p>Should not render</p>
      </PageContainer>,
    )
    expect(screen.queryByText('Should not render')).not.toBeInTheDocument()
    expect(container.querySelector('.sl-page')).toBeInTheDocument()
  })

  it('shows custom loadingContent when provided', () => {
    render(
      <PageContainer title="Settings" loading loadingContent={<div data-testid="custom-loading">Loading...</div>}>
        <p>Should not render</p>
      </PageContainer>,
    )
    expect(screen.getByTestId('custom-loading')).toBeInTheDocument()
    expect(screen.queryByText('Should not render')).not.toBeInTheDocument()
  })

  it('renders header with custom loadingContent', () => {
    render(
      <PageContainer title="Settings" loading loadingContent={<div>Custom</div>}>
        <p>Content</p>
      </PageContainer>,
    )
    expect(screen.getByText('Settings')).toBeInTheDocument()
    expect(screen.getByText('Custom')).toBeInTheDocument()
  })

  // ── Error ────────────────────────────────────────────

  it('shows error state with message', () => {
    render(
      <PageContainer title="Settings" error="Failed to load data">
        <p>Hidden content</p>
      </PageContainer>,
    )
    expect(screen.getByText('Failed to load data')).toBeInTheDocument()
    expect(screen.queryByText('Hidden content')).not.toBeInTheDocument()
  })

  it('shows retry button when error + onRetry provided', () => {
    const onRetry = vi.fn()
    render(
      <PageContainer title="Settings" error="Something went wrong" onRetry={onRetry}>
        <p>Content</p>
      </PageContainer>,
    )
    const retryBtn = screen.getByRole('button', { name: /retry/i })
    expect(retryBtn).toBeInTheDocument()
    retryBtn.click()
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('hides retry button when onRetry not provided', () => {
    render(
      <PageContainer title="Settings" error="Error">
        <p>Content</p>
      </PageContainer>,
    )
    expect(screen.queryByRole('button', { name: /retry/i })).not.toBeInTheDocument()
  })

  // ── Empty ────────────────────────────────────────────

  it('shows empty state when empty=true', () => {
    render(
      <PageContainer title="Knowledge" empty emptyMessage="No items found">
        <p>Should not render</p>
      </PageContainer>,
    )
    expect(screen.getByText('No items found')).toBeInTheDocument()
    expect(screen.queryByText('Should not render')).not.toBeInTheDocument()
  })

  it('renders header with empty state', () => {
    render(
      <PageContainer title="Knowledge" empty emptyMessage="Empty">
        <p>Content</p>
      </PageContainer>,
    )
    expect(screen.getByText('Knowledge')).toBeInTheDocument()
    expect(screen.getByText('Empty')).toBeInTheDocument()
  })

  it('renders toolbar with empty state', () => {
    render(
      <PageContainer
        title="Knowledge"
        empty
        emptyMessage="Empty"
        toolbar={<input placeholder="Search..." />}
      >
        <p>Content</p>
      </PageContainer>,
    )
    expect(screen.getByPlaceholderText('Search...')).toBeInTheDocument()
  })

  // ── Toolbar ──────────────────────────────────────────

  it('renders toolbar between header and content', () => {
    const { container } = render(
      <PageContainer title="Datasets" toolbar={<input placeholder="Search datasets..." />}>
        <p>Content</p>
      </PageContainer>,
    )
    expect(screen.getByPlaceholderText('Search datasets...')).toBeInTheDocument()
    expect(screen.getByText('Content')).toBeInTheDocument()
    const toolbar = container.querySelector('.mb-4')
    expect(toolbar).toBeInTheDocument()
  })

  it('hides toolbar when not provided', () => {
    const { container } = render(
      <PageContainer title="Settings">
        <p>Content</p>
      </PageContainer>,
    )
    expect(container.querySelector('.mb-4')).not.toBeInTheDocument()
  })

  // ── Styling ──────────────────────────────────────────

  it('applies custom className to outer wrapper', () => {
    const { container } = render(
      <PageContainer title="Settings" className="custom-class">
        <p>Content</p>
      </PageContainer>,
    )
    expect(container.querySelector('.sl-page')?.classList.contains('custom-class')).toBe(true)
  })

  it('applies contentClassName to content wrapper', () => {
    const { container } = render(
      <PageContainer title="Settings" contentClassName="custom-content">
        <p>Content</p>
      </PageContainer>,
    )
    expect(container.querySelector('.custom-content')).toBeInTheDocument()
  })

  it('applies className to error state', () => {
    const { container } = render(
      <PageContainer title="Settings" error="Error" className="custom-class">
        <p>Content</p>
      </PageContainer>,
    )
    expect(container.querySelector('.sl-page')?.classList.contains('custom-class')).toBe(true)
  })

  it('applies className to empty state', () => {
    const { container } = render(
      <PageContainer title="Settings" empty emptyMessage="Empty" className="custom-class">
        <p>Content</p>
      </PageContainer>,
    )
    expect(container.querySelector('.sl-page')?.classList.contains('custom-class')).toBe(true)
  })

  it('applies className to loading state with loadingContent', () => {
    const { container } = render(
      <PageContainer title="Settings" loading loadingContent={<div />} className="custom-class">
        <p>Content</p>
      </PageContainer>,
    )
    expect(container.querySelector('.sl-page')?.classList.contains('custom-class')).toBe(true)
  })
})
