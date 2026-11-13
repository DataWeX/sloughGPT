'use client'

import { chatDB, type QuickPrompt as DBQuickPrompt } from '@/lib/db'

export type QuickPrompt = DBQuickPrompt

interface PromptEntry {
  name: string
  description: string
  prompt: string
  icon: string
  category: QuickPrompt['category']
}

const DEFAULT_PROMPTS: PromptEntry[] = [
  { name: 'Explain simply', description: 'Explain like I am 5', icon: '💡', category: 'learning', prompt: 'Explain this in simple, easy-to-understand terms:\n\n{{text}}' },
  { name: 'Summarize', description: 'Condense key points', icon: '📝', category: 'writing', prompt: 'Summarize the following concisely, capturing the key points:\n\n{{text}}' },
  { name: 'Brainstorm', description: 'Creative idea generation', icon: '🧠', category: 'planning', prompt: 'Brainstorm creative ideas and approaches for:\n\n{{text}}' },
  { name: 'TLDR', description: 'One-sentence summary', icon: '⚡', category: 'learning', prompt: 'Give me a one-sentence TLDR of the following:\n\n{{text}}' },
  { name: 'Make a plan', description: 'Step-by-step plan', icon: '📋', category: 'planning', prompt: 'Create a detailed step-by-step plan for:\n\n{{text}}' },
  { name: 'Code review', description: 'Review for issues', icon: '🔍', category: 'coding', prompt: 'Review this code for bugs, security issues, and style problems:\n\n{{text}}' },
  { name: 'Write a poem', description: 'Creative verse', icon: '🎭', category: 'writing', prompt: 'Write a creative poem about:\n\n{{text}}' },
  { name: 'Improve writing', description: 'Polish and refine', icon: '✏️', category: 'writing', prompt: 'Improve this writing for clarity, flow, and impact:\n\n{{text}}' },
  { name: 'Debate both sides', description: 'Pros and cons', icon: '⚖️', category: 'learning', prompt: 'Present a balanced analysis of both sides of:\n\n{{text}}' },
  { name: 'Refactor code', description: 'Clean up code', icon: '♻️', category: 'coding', prompt: 'Refactor this code to be cleaner, more maintainable, and follow best practices:\n\n{{text}}' },
  { name: 'Write tests', description: 'Generate test cases', icon: '🧪', category: 'coding', prompt: 'Write comprehensive test cases for:\n\n{{text}}' },
  { name: 'Study guide', description: 'Learn this topic', icon: '📚', category: 'learning', prompt: 'Create a structured study guide covering the key concepts of:\n\n{{text}}' },
]

function applyPrompt(template: string, text: string): string {
  return template.replace(/\{\{text\}\}/g, text)
}

let cached: QuickPrompt[] | null = null

async function ensureInit(): Promise<QuickPrompt[]> {
  if (cached) return cached
  const stored = await chatDB.getPrompts()
  if (stored.length > 0) {
    cached = stored
    return cached
  }
  const defaults: QuickPrompt[] = DEFAULT_PROMPTS.map((p, i) => ({
    id: `default-${i}`,
    ...p,
    createdAt: 0,
    updatedAt: 0,
  }))
  await chatDB.importPrompts(defaults)
  cached = defaults
  return cached
}

async function initPrompts(): Promise<void> {
  cached = null
  await ensureInit()
}

export async function listPrompts(): Promise<QuickPrompt[]> {
  return ensureInit()
}

export async function listPromptsByCategory(): Promise<Record<string, QuickPrompt[]>> {
  const all = await ensureInit()
  const grouped: Record<string, QuickPrompt[]> = {}
  for (const p of all) {
    if (!grouped[p.category]) grouped[p.category] = []
    grouped[p.category].push(p)
  }
  return grouped
}

export async function getPrompt(id: string): Promise<QuickPrompt | undefined> {
  const all = await ensureInit()
  return all.find(p => p.id === id)
}

export async function createPrompt(entry: Omit<QuickPrompt, 'id' | 'createdAt' | 'updatedAt'>): Promise<QuickPrompt> {
  const prompt: QuickPrompt = {
    ...entry,
    id: `prompt_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }
  await chatDB.savePrompt(prompt)
  cached = null
  return prompt
}

export async function updatePrompt(id: string, updates: Partial<Omit<QuickPrompt, 'id' | 'createdAt'>>): Promise<QuickPrompt | undefined> {
  const all = await ensureInit()
  const idx = all.findIndex(p => p.id === id)
  if (idx === -1) return undefined
  all[idx] = { ...all[idx], ...updates, updatedAt: Date.now() }
  await chatDB.savePrompt(all[idx])
  cached = all
  return all[idx]
}

export async function deletePrompt(id: string): Promise<boolean> {
  const all = await ensureInit()
  const filtered = all.filter(p => p.id !== id)
  if (filtered.length === all.length) return false
  await chatDB.deletePrompt(id)
  cached = filtered
  return true
}

export async function resetToDefaults(): Promise<QuickPrompt[]> {
  await chatDB.clearPrompts()
  cached = null
  return ensureInit()
}

export { applyPrompt, initPrompts }

/** Reset module-level cache. For test use only. */
export function __resetPromptCache(): void {
  cached = null
}
