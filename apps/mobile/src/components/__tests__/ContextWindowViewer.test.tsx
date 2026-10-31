import React from 'react'
import { render, fireEvent } from '@testing-library/react-native'
import { TamaguiProvider } from 'tamagui'
import appConfig from '../../../tamagui.config'
import { ContextWindowViewer, type ContextWindowItem } from '../ContextWindowViewer'
import * as clipboardService from '../../services/clipboard'

jest.mock('../../services/clipboard', () => ({
  copyToClipboard: jest.fn(async () => true),
}))

const Wrapper = ({ children }: any) => (
  <TamaguiProvider config={appConfig} defaultTheme="light">
    {children}
  </TamaguiProvider>
)

const items: ContextWindowItem[] = [
  { type: 'system', label: 'System Prompt', content: 'You are helpful.', tokenCount: 5 },
  { type: 'user', label: 'User Messages', content: 'hi there' },
]

async function renderViewer(props: Partial<React.ComponentProps<typeof ContextWindowViewer>> = {}) {
  return render(<ContextWindowViewer items={items} {...props} />, { wrapper: Wrapper })
}

describe('ContextWindowViewer', () => {
  it('shows empty state when no items', async () => {
    const { getByText } = await renderViewer({ items: [] })
    expect(getByText('No context items loaded')).toBeTruthy()
  })

  it('renders header with item count and token estimate', async () => {
    const { getByText } = await renderViewer()
    expect(getByText('Context Window')).toBeTruthy()
    expect(getByText(/2 items/)).toBeTruthy()
    expect(getByText(/items · ~/)).toBeTruthy()
  })

  it('shows type badges and labels', async () => {
    const { getAllByText } = await renderViewer()
    expect(getAllByText('System Prompt').length).toBeGreaterThanOrEqual(2)
    expect(getAllByText('User Messages').length).toBeGreaterThanOrEqual(1)
  })

  it('toggles item expansion on press', async () => {
    const { getByText, getAllByText, queryByText } = await renderViewer()
    expect(queryByText('You are helpful.')).toBeNull()
    await fireEvent.press(getAllByText('System Prompt')[0])
    expect(getByText('You are helpful.')).toBeTruthy()
    await fireEvent.press(getAllByText('System Prompt')[0])
    expect(queryByText('You are helpful.')).toBeNull()
  })

  it('Expand All reveals all contents, Collapse All hides them', async () => {
    const { getByText, getAllByText, queryByText } = await renderViewer()
    await fireEvent.press(getByText('Expand All'))
    expect(getByText('You are helpful.')).toBeTruthy()
    expect(getByText('hi there')).toBeTruthy()
    await fireEvent.press(getByText('Collapse All'))
    expect(queryByText('You are helpful.')).toBeNull()
  })

  it('copies item content when copy pressed', async () => {
    const { getAllByText, getByTestId } = await renderViewer()
    await fireEvent.press(getAllByText('System Prompt')[0])
    await fireEvent.press(getByTestId('copy-context-item'))
    expect(clipboardService.copyToClipboard).toHaveBeenCalledWith('You are helpful.')
  })

  it('calls onRefresh when refresh pressed', async () => {
    const onRefresh = jest.fn()
    const { getByText } = await renderViewer({ onRefresh })
    await fireEvent.press(getByText('Refresh'))
    expect(onRefresh).toHaveBeenCalled()
  })
})
