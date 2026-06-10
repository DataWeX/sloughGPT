'use client'

import { useState, useCallback } from 'react'
import type { AgentDef } from '@/lib/agents'

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
    localStorage.setItem('man_current_agent', a.id)
  }, [])

  const handleToggleKnowledge = useCallback(() => {
    setKnowledgeCtx(prev => ({ ...prev, showing: !prev.showing }))
  }, [])

  return {
    agents, setAgents,
    currentAgent, setCurrentAgent,
    knowledgeCtx, setKnowledgeCtx,
    handleSelectAgent,
    handleToggleKnowledge,
  }
}
