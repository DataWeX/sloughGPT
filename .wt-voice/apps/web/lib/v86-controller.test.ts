import { describe, it, expect, vi, beforeEach } from 'vitest'
import { V86Controller } from './v86-controller'

describe('V86Controller', () => {
  let controller: V86Controller

  beforeEach(() => {
    controller = new V86Controller()
  })

  it('creates instance with no emulator', () => {
    expect(controller.isRunning()).toBe(false)
  })

  it('throws on saveState when not initialized', async () => {
    await expect(controller.saveState()).rejects.toThrow('Emulator not initialized')
  })

  it('throws on restoreState when not initialized', async () => {
    const buf = new ArrayBuffer(8)
    await expect(controller.restoreState(buf)).rejects.toThrow('Emulator not initialized')
  })

  it('destroy is safe when not initialized', () => {
    expect(() => controller.destroy()).not.toThrow()
  })

  it('restart is safe when not initialized', () => {
    expect(() => controller.restart()).not.toThrow()
  })
})
