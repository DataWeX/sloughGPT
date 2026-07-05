import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('./VisionStudioDialog', () => ({
  VisionStudioDialog: (props: any) => (
    <div data-testid="vision-studio-dialog" data-open={props.open} data-session-id={props.sessionId}>
      VisionStudioDialog
    </div>
  ),
}))

import { VisionTabContent } from './VisionTabContent'

describe('VisionTabContent', () => {
  const onGeneratedImage = vi.fn()
  const onSendText = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders Vision Model header', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    expect(screen.getByText('Vision Model')).toBeDefined()
  })

  it('shows images learned count defaulting to 0', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    expect(screen.getByText('0')).toBeDefined()
  })

  it('shows images learned count from props', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
        visionImagesLearned={42}
      />
    )
    expect(screen.getByText('42')).toBeDefined()
  })

  it('shows status from visionStatus prop', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
        visionStatus="active"
      />
    )
    expect(screen.getByText(/active/)).toBeDefined()
  })

  it('shows Trained label when visionTrained is true', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
        visionTrained={true}
        visionImagesLearned={10}
      />
    )
    expect(screen.getByText(/Trained/)).toBeDefined()
  })

  it('shows Learning label when images learned > 0 but not trained', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
        visionImagesLearned={5}
        visionTrained={false}
      />
    )
    expect(screen.getByText(/Learning/)).toBeDefined()
  })

  it('shows Ready label when no training data exists', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    expect(screen.getByText(/Ready/)).toBeDefined()
  })

  it('shows mean accuracy when > 0', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
        meanAccuracy={75}
      />
    )
    expect(screen.getByText('Mean accuracy')).toBeDefined()
    expect(screen.getByText('75.0%')).toBeDefined()
  })

  it('does not show mean accuracy when 0', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
        meanAccuracy={0}
      />
    )
    expect(screen.queryByText('Mean accuracy')).toBeNull()
  })

  it('does not show mean accuracy when undefined', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    expect(screen.queryByText('Mean accuracy')).toBeNull()
  })

  it('shows vocabulary when > 0', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
        visionVocabSize={256}
      />
    )
    expect(screen.getByText('Vocabulary')).toBeDefined()
    expect(screen.getByText('256 words')).toBeDefined()
  })

  it('does not show vocabulary when 0', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
        visionVocabSize={0}
      />
    )
    expect(screen.queryByText('Vocabulary')).toBeNull()
  })

  it('opens VisionStudioDialog on button click', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    const dialog = screen.getByTestId('vision-studio-dialog')
    expect(dialog.getAttribute('data-open')).toBe('false')
    fireEvent.click(screen.getByText('Open Vision Studio'))
    expect(dialog.getAttribute('data-open')).toBe('true')
  })

  it('passes sessionId to VisionStudioDialog', () => {
    render(
      <VisionTabContent
        sessionId="test-session-123"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    const dialog = screen.getByTestId('vision-studio-dialog')
    expect(dialog.getAttribute('data-session-id')).toBe('test-session-123')
  })

  it('passes initialCaps to VisionStudioDialog via VisionTabContent', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
        visionImagesLearned={15}
        visionTrained={true}
        visionVocabSize={512}
        meanAccuracy={88}
      />
    )
    expect(screen.getByText('15')).toBeDefined()
    expect(screen.getByText(/Trained/)).toBeDefined()
    expect(screen.getByText('88.0%')).toBeDefined()
    expect(screen.getByText('512 words')).toBeDefined()
  })

  it('renders Open Vision Studio button', () => {
    render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    expect(screen.getByText('Open Vision Studio')).toBeDefined()
  })

  it('has a status dot element', () => {
    const { container } = render(
      <VisionTabContent
        sessionId="s1"
        onGeneratedImage={onGeneratedImage}
        onSendText={onSendText}
      />
    )
    const dots = container.querySelectorAll('.rounded-full')
    expect(dots.length).toBeGreaterThanOrEqual(1)
  })
})
