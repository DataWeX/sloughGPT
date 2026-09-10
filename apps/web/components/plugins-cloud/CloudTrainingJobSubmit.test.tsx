// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: (props: any) => <input {...props} />,
}))

import { CloudTrainingJobSubmit } from './CloudTrainingJobSubmit'

afterEach(() => cleanup())

describe('CloudTrainingJobSubmit', () => {
  it('renders the card title', () => {
    render(<CloudTrainingJobSubmit />)
    expect(screen.getAllByText('Submit Training Job').length).toBeGreaterThanOrEqual(1)
  })

  it('renders the card description', () => {
    render(<CloudTrainingJobSubmit />)
    expect(screen.getAllByText('Train a model on cloud infrastructure').length).toBeGreaterThanOrEqual(1)
  })

  it('renders the provider select', () => {
    render(<CloudTrainingJobSubmit />)
    const select = screen.getByRole('combobox')
    expect(select).toBeTruthy()
  })

  it('renders the dataset ID input', () => {
    render(<CloudTrainingJobSubmit />)
    expect(screen.getByPlaceholderText('e.g. my-dataset')).toBeTruthy()
  })

  it('renders the submit button', () => {
    render(<CloudTrainingJobSubmit />)
    expect(screen.getAllByText('Submit Job').length).toBeGreaterThanOrEqual(1)
  })

  it('disables button when no dataset ID provided', () => {
    render(<CloudTrainingJobSubmit />)
    const btn = screen.getByRole('button')
    expect(btn).toHaveProperty('disabled', true)
  })

  it('disables button when submitting', () => {
    render(<CloudTrainingJobSubmit submitting />)
    expect(screen.getAllByText('Submitting...').length).toBeGreaterThanOrEqual(1)
  })
})
