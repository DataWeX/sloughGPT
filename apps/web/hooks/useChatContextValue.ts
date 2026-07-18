'use client'

import { useMemo, useCallback } from 'react'
import type { ChatHealthContextValue, ChatModelContextValue, ChatUICallbacksContextValue } from '@/contexts/ChatContext'
import { soulsController } from '@/lib/souls-controller'
import { apiPost } from '@/lib/http-client'
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

export function useChatHealthValue(opts: Pick<UseChatContextValueOpts, 'health' | 'refreshHealth'>): ChatHealthContextValue {
  return useMemo(() => ({
    health: opts.health,
    refreshHealth: opts.refreshHealth,
  }), [opts.health, opts.refreshHealth])
}

export function useChatModelValue(opts: Pick<UseChatContextValueOpts, 'model' | 'agents' | 'vision' | 'chat' | 'showToast'>): ChatModelContextValue {
  const { model, agents, vision, chat, showToast } = opts

  const onLoadCheckpoint = useCallback(async (name: string) => {
    try {
      await soulsController.loadCheckpoint(name)
      model.setCurrentCheckpoint(name)
      showToast(`Trained version loaded: ${name}`)
    } catch { showToast('Failed to load trained version', 'error') }
  }, [model.setCurrentCheckpoint, showToast])

  const onTrainStep = useCallback(async () => {
    model.setLearnerTraining(true)
    try {
      const data = await apiPost<{ current_loss?: number; train_steps_completed?: number; loss_history?: Array<{ step: number; loss: number; tokens: number; timestamp: number }> }>('/learn/train', {})
      if (data.current_loss !== undefined) showToast(`Training step: loss ${data.current_loss.toFixed(4)}`)
      else showToast('Training step complete')
      model.setLearnerInfo(prev => {
        if (!prev) return prev
        return {
          ...prev,
          train_steps_completed: data.train_steps_completed ?? prev.train_steps_completed,
          current_loss: data.current_loss ?? prev.current_loss,
          loss_history: data.loss_history ?? prev.loss_history,
        }
      })
    } catch { showToast('Training step failed', 'error') }
    finally { model.setLearnerTraining(false) }
  }, [model.setLearnerTraining, model.setLearnerInfo, showToast])

  return useMemo<ChatModelContextValue>(() => ({
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
    setInput: chat.setInput,
  }), [model, agents, vision, chat.setInput, showToast, onLoadCheckpoint, onTrainStep])
}

export function useChatUIValue(opts: Pick<UseChatContextValueOpts, 'ui' | 'showToast'>): ChatUICallbacksContextValue {
  return useMemo(() => ({
    onOpenSettings: opts.ui.toggleSettings,
    onOpenShortcuts: () => window.dispatchEvent(new CustomEvent('toggle-shortcuts')),
    onOpenConversationViewer: () => opts.ui.setShowConversationViewer(true),
    showToast: opts.showToast,
  }), [opts.ui.toggleSettings, opts.ui.setShowConversationViewer, opts.showToast])
}
