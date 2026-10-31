/**
 * Maps GET /context/inspect (ContextCore.get_context_inspector) to
 * ContextWindowItem[] for the ContextWindowViewer — pure, no I/O.
 */

export interface ContextWindowItem {
  label: string
  content: string
  tokenCount?: number
  type: 'system' | 'knowledge' | 'agent' | 'user' | 'assistant'
}

export interface ContextInspectorMessage {
  role: string
  content: string
}

export interface ContextInspector {
  system_prompt?: string
  session_messages?: ContextInspectorMessage[]
  working_memory?: unknown
  semantic_keys?: string[]
  episodic_count?: number
  sensory_buffer_size?: number
}

export function estimateTokens(items: ContextWindowItem[], totalTokens?: number): number {
  if (totalTokens !== undefined) {
    return totalTokens
  }
  return items.reduce(
    (sum, item) => sum + (item.tokenCount ?? Math.ceil(item.content.length / 4)),
    0,
  )
}

function stringifyContent(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) {
    return value.map((v) => (typeof v === 'string' ? v : JSON.stringify(v))).join('\n')
  }
  if (value === null || value === undefined) return ''
  return JSON.stringify(value, null, 2)
}

export function contextItemsFromInspector(inspector: ContextInspector): ContextWindowItem[] {
  const items: ContextWindowItem[] = []

  const systemPrompt = (inspector.system_prompt || '').trim()
  if (systemPrompt) {
    items.push({
      type: 'system',
      label: 'System Prompt',
      content: systemPrompt,
      tokenCount: Math.ceil(systemPrompt.length / 4),
    })
  }

  const messages = inspector.session_messages || []
  const byRole = (role: string) => messages.filter((m) => m.role === role)
  for (const [role, type, label] of [
    ['user', 'user', 'User Messages'],
    ['assistant', 'assistant', 'Assistant Messages'],
  ] as const) {
    const content = byRole(role)
      .map((m) => m.content)
      .filter(Boolean)
      .join('\n\n')
      .trim()
    if (content) {
      items.push({
        type,
        label,
        content,
        tokenCount: Math.ceil(content.length / 4),
      })
    }
  }

  const semantic = (inspector.semantic_keys || []).filter(Boolean)
  if (semantic.length > 0) {
    items.push({
      type: 'knowledge',
      label: 'Semantic Memory',
      content: semantic.join(', '),
      tokenCount: Math.ceil(semantic.join(', ').length / 4),
    })
  }

  const episodic = inspector.episodic_count || 0
  if (episodic > 0) {
    items.push({
      type: 'knowledge',
      label: 'Episodic Memory',
      content: `${episodic} stored episodes`,
    })
  }

  const working = stringifyContent(inspector.working_memory).trim()
  if (working) {
    items.push({
      type: 'agent',
      label: 'Working Memory',
      content: working,
      tokenCount: Math.ceil(working.length / 4),
    })
  }

  return items
}
