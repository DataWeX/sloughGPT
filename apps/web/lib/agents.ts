// Single source of truth for agents
// Used by ChatHeader and chat page

export const AGENTS = {
  general: { name: 'General', prompt: 'You are a helpful AI assistant.' },
  coder: { name: 'Coder', prompt: 'You are an expert programmer. Write clean code with comments.' },
  writer: { name: 'Writer', prompt: 'You are a creative writing assistant.' },
  researcher: { name: 'Researcher', prompt: 'Be thorough and cite sources when available.' },
} as const

export type AgentId = keyof typeof AGENTS

export function getAgentName(agentId: string): string {
  return AGENTS[agentId as AgentId]?.name ?? 'General'
}