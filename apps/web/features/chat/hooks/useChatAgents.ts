'use client'

import { useState, useCallback } from 'react'
import { agentsController } from '@/lib/agents-controller'
import { AGENTS } from '@/lib/agents'
import type { AgentDef } from '@/lib/agents'
import { chatDB } from '@/lib/db'

export function useChatAgents() {
  const [agents, setAgents] = useState<AgentDef[]>([])
  const [currentAgent, setCurrentAgent] = useState<AgentDef | null>(null)
  const [knowledgeCtx, setKnowledgeCtx] = useState<{
    showing: boolean
    count: number
    context: string
  }>({ showing: false, count: 0, context: '' })

  const handleSelectAgent = useCallback((a: AgentDef) => {
    setCurrentAgent(a)
    chatDB.setKV('man_current_agent', a.id)
  }, [])

  const handleToggleKnowledge = useCallback(() => {
    setKnowledgeCtx(prev => ({ ...prev, showing: !prev.showing }))
  }, [])

  const fetchInitialData = useCallback(async () => {
    try {
      const data = await agentsController.list()
      const localAgents = Object.values(AGENTS)
      const merged = data && data.length > 0 ? data : localAgents
      setAgents(merged)
      const savedAgentId = await chatDB.getKV<string>('man_current_agent') || 'general'
      const found = merged.find(a => a.id === savedAgentId)
      if (found) setCurrentAgent(found)
    } catch {
      const localAgents = Object.values(AGENTS)
      setAgents(localAgents)
      const savedAgentId = await chatDB.getKV<string>('man_current_agent') || 'general'
      const found = localAgents.find(a => a.id === savedAgentId)
      if (found) setCurrentAgent(found)
    }
  }, [setAgents, setCurrentAgent])

  return {
    agents, setAgents,
    currentAgent, setCurrentAgent,
    knowledgeCtx, setKnowledgeCtx,
    handleSelectAgent,
    handleToggleKnowledge,
    fetchInitialData,
  }
}
