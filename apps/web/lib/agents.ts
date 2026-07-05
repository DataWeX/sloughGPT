// Agents define the AI's role, expertise, and behavior patterns
// Distinct from Models (the brain) and Souls (the personality)
//
// Model  = Which neural network generates tokens
// Soul   = Emotional/behavioral traits (warmth, creativity, etc.)
// Agent  = Role/expertise + system instructions + capabilities

export interface AgentDef {
  id: string
  name: string
  description: string
  instructions: string
  icon?: string
  capabilities?: string[]
}

export const AGENTS: Record<string, AgentDef> = {
  general: {
    id: 'general',
    name: 'General',
    description: 'Helpful AI assistant for everyday tasks',
    instructions: 'You are a helpful AI assistant. Be clear, concise, and friendly.',
    icon: '💬',
    capabilities: ['conversation', 'questions', 'general-knowledge'],
  },
  coder: {
    id: 'coder',
    name: 'Coder',
    description: 'Expert programmer for code review and development',
    instructions: 'You are an expert programmer. Write clean, well-documented code. Explain your reasoning. Follow best practices for the language.',
    icon: '💻',
    capabilities: ['code-generation', 'debugging', 'code-review', 'architecture'],
  },
  writer: {
    id: 'writer',
    name: 'Writer',
    description: 'Creative writing and content creation',
    instructions: 'You are a creative writing assistant. Help with storytelling, editing, and content creation. Be imaginative and engaging.',
    icon: '✍️',
    capabilities: ['creative-writing', 'editing', 'storytelling', 'content-strategy'],
  },
  researcher: {
    id: 'researcher',
    name: 'Researcher',
    description: 'Thorough analysis and fact-based responses',
    instructions: 'You are a research assistant. Be thorough, cite sources when available, distinguish facts from opinions, and acknowledge uncertainty.',
    icon: '🔬',
    capabilities: ['analysis', 'fact-checking', 'summarization', 'citation'],
  },
  analyst: {
    id: 'analyst',
    name: 'Analyst',
    description: 'Data analysis and structured thinking',
    instructions: 'You are a data analyst. Break down complex problems, use structured reasoning, and present findings clearly with supporting evidence.',
    icon: '📊',
    capabilities: ['data-analysis', 'structured-reasoning', 'visualization', 'reporting'],
  },
} as const

export type AgentId = keyof typeof AGENTS

export function getAgentName(agentId: string): string {
  return AGENTS[agentId as AgentId]?.name ?? 'General'
}

export function getAgentDescription(agentId: string): string {
  return AGENTS[agentId as AgentId]?.description ?? ''
}

export function getAgentSystemPrompt(agentId: string): string {
  return AGENTS[agentId as AgentId]?.instructions ?? AGENTS.general.instructions
}

export function getAgentCapabilities(agentId: string): string[] {
  return AGENTS[agentId as AgentId]?.capabilities ?? []
}
