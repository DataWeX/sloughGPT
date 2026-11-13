/**
 * Companion Controller — axios-based API for AI companion management.
 *
 * Usage:
 *   import { companionController } from '@/lib/companion-controller'
 *   const info = await companionController.getInfo()
 *   await companionController.setPreset('warm')
 */

import { apiGet, apiPost, apiPatch, apiDelete } from './http-client'

export interface CompanionTraits {
  name: string
  warmth: number
  curiosity: number
  creativity: number
  confidence: number
  humor: number
}

export interface CompanionInfo {
  traits: CompanionTraits
  mood?: string
  system_prompt?: string
}

export interface CompanionPreset {
  id: string
  name: string
  description: string
}

export const companionController = {
  async getInfo(): Promise<CompanionInfo> {
    return apiGet<CompanionInfo>('/companion/')
  },

  async setPersonality(traits: CompanionTraits): Promise<{ status: string; traits: CompanionTraits }> {
    return apiPost('/companion/personality', traits)
  },

  async patchPersonality(partial: Partial<CompanionTraits>): Promise<{ status: string; traits: CompanionTraits }> {
    return apiPatch('/companion/personality', partial)
  },

  async setPreset(preset: string, name?: string): Promise<{ status: string; preset: string; traits: CompanionTraits }> {
    return apiPost('/companion/preset', { preset, name: name ?? 'Friend' })
  },

  async getPrompt(): Promise<{ system_prompt: string }> {
    return apiGet<{ system_prompt: string }>('/companion/prompt')
  },

  async listPresets(): Promise<{ presets: CompanionPreset[] }> {
    return apiGet<{ presets: CompanionPreset[] }>('/companion/presets')
  },

  async reset(): Promise<{ status: string; traits: CompanionTraits }> {
    return apiDelete('/companion/')
  },

  async chat(message: string, opts?: { user_name?: string; user_mood?: string }): Promise<{ response: string; system_prompt: string }> {
    return apiPost('/companion/chat', {
      message,
      user_name: opts?.user_name,
      user_mood: opts?.user_mood,
      include_system_prompt: true,
    })
  },
}
