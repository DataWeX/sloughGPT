import { describe, it, expect, beforeEach } from 'vitest'
import { useSettingsStore } from '@/stores/settings-store'

describe('Settings Store', () => {
  beforeEach(() => {
    useSettingsStore.getState().reset()
  })

  it('should initialize with default values', () => {
    const state = useSettingsStore.getState()
    expect(state.theme).toBe('system')
    expect(state.temperature).toBe(0.8)
    expect(state.maxTokens).toBe(256)
    expect(state.memoryContext).toBe('')
    expect(state.apiUrl).toBe('')
  })

  it('should set theme', () => {
    useSettingsStore.getState().setTheme('dark')
    expect(useSettingsStore.getState().theme).toBe('dark')

    useSettingsStore.getState().setTheme('light')
    expect(useSettingsStore.getState().theme).toBe('light')
  })

  it('should update multiple settings', () => {
    useSettingsStore.getState().update({
      temperature: 1.2,
      maxTokens: 512,
      memoryContext: 'Test context',
    })

    const state = useSettingsStore.getState()
    expect(state.temperature).toBe(1.2)
    expect(state.maxTokens).toBe(512)
    expect(state.memoryContext).toBe('Test context')
  })

  it('should reset to defaults', () => {
    // Change settings
    useSettingsStore.getState().update({
      theme: 'dark',
      temperature: 1.5,
      maxTokens: 1024,
    })

    // Reset
    useSettingsStore.getState().reset()

    const state = useSettingsStore.getState()
    expect(state.theme).toBe('system')
    expect(state.temperature).toBe(0.8)
    expect(state.maxTokens).toBe(256)
  })

  it('should update apiUrl', () => {
    useSettingsStore.getState().update({ apiUrl: 'http://localhost:8000' })
    expect(useSettingsStore.getState().apiUrl).toBe('http://localhost:8000')
  })
})
