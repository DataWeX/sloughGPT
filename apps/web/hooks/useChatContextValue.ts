'use client'

import { useMemo, useCallback } from 'react'
import type { ChatContextValue } from '@/contexts/ChatContext'
import { soulsController } from '@/lib/souls-controller'
import { PUBLIC_API_URL } from '@/lib/config'
import type { useChatUI } from './useChatUI'
import type { useChatVision } from './useChatVision'
import type { useChatAgents } from './useChatAgents'
import type { useChatModelSettings } from './useChatModelSettings'
import type { useChatMessages } from './useChatMessages'
import type { ApiHealthSnapshot } from './useApiHealth'

interface UseChatContextValueOpts {
  health: ApiHealthSnapshot
  refreshHealth: () => Promise<void>
  model: ReturnType<typeof useChatModelSettings>
  agents: ReturnType<typeof useChatAgents>
  vision: ReturnType<typeof useChatVision>
  ui: ReturnType<typeof useChatUI>
  chat: ReturnType<typeof useChatMessages>
  showToast: (message: string, type?: string) => void
}

export function useChatContextValue(opts: UseChatContextValueOpts) {
  const { health, refreshHealth, model, agents, vision, ui, chat, showToast } = opts

  const onLoadCheckpoint = useCallback(async (name: string) => {
    try {
      await soulsController.loadCheckpoint(name)
      model.setCurrentCheckpoint(name)
      showToast(`Checkpoint loaded: ${name}`)
    } catch { showToast('Failed to load checkpoint', 'error') }
  }, [model.setCurrentCheckpoint, showToast])

  const onTrainStep = useCallback(async () => {
    model.setLearnerTraining(true)
    try {
      const resp = await fetch(`${PUBLIC_API_URL}/learn/train`, { method: 'POST' })
      if (resp.ok) {
        const data = await resp.json()
        if (data.current_loss !== undefined) showToast(`Train step: loss ${data.current_loss.toFixed(4)}`)
        else showToast('Train step complete')
        model.setLearnerInfo(prev => {
          if (!prev) return prev
          return {
            ...prev,
            train_steps_completed: data.train_steps_completed ?? prev.train_steps_completed,
            current_loss: data.current_loss ?? prev.current_loss,
            loss_history: data.loss_history ?? prev.loss_history,
          }
        })
      } else showToast('Train step failed', 'error')
    } catch { showToast('Train step failed', 'error') }
    finally { model.setLearnerTraining(false) }
  }, [model.setLearnerTraining, model.setLearnerInfo, showToast])

  return useMemo<ChatContextValue>(() => ({
    health,
    refreshHealth,
    model: model.model,
    setModel: model.setModel,
    availableModels: model.availableModels,
    modelInfoMap: model.modelInfoMap,
    temperature: model.temperature,
    setTemperature: model.setTemperature,
    maxTokens: model.maxTokens,
    setMaxTokens: model.setMaxTokens,
    loadingModel: model.loadingModel,
    handleSelectModel: model.handleSelectModel,
    handleUnloadModel: model.handleUnloadModel,
    souls: model.souls,
    currentSoul: model.currentSoul,
    setCurrentSoul: model.setCurrentSoul,
    handleSelectSoul: model.handleSelectSoul,
    checkpoints: model.checkpoints,
    currentCheckpoint: model.currentCheckpoint,
    setCurrentCheckpoint: model.setCurrentCheckpoint,
    onLoadCheckpoint,
    agents: agents.agents,
    currentAgent: agents.currentAgent,
    setCurrentAgent: agents.setCurrentAgent,
    visionCaps: vision.visionCaps,
    visionCaptionHistory: vision.visionCaptionHistory,
    visionVocabSize: vision.visionVocabSize,
    learnerInfo: model.learnerInfo,
    learnerTraining: model.learnerTraining,
    setLearnerInfo: model.setLearnerInfo,
    setLearnerTraining: model.setLearnerTraining,
    onTrainStep,
    onOpenSettings: ui.toggleSettings,
    onOpenShortcuts: () => window.dispatchEvent(new CustomEvent('toggle-shortcuts')),
    onOpenConversationViewer: () => ui.setShowConversationViewer(true),
    setInput: chat.setInput,
    showToast,
  }), [health, refreshHealth, model, agents, vision, ui, chat.setInput, showToast, onLoadCheckpoint, onTrainStep])
}
