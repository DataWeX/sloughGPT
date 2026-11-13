import { z } from 'zod'

export const knowledgeSchema = z.object({
  content: z.string().min(1, 'Content is required').max(10000, 'Content must be under 10,000 characters'),
  topic: z.string().max(100, 'Topic must be under 100 characters').optional().or(z.literal('')),
})

export type KnowledgeFormData = z.infer<typeof knowledgeSchema>

export const agentSchema = z.object({
  name: z.string().min(1, 'Name is required').max(50, 'Name must be under 50 characters'),
  description: z.string().max(200, 'Description must be under 200 characters').optional().or(z.literal('')),
  instructions: z.string().max(5000, 'Instructions must be under 5,000 characters').optional().or(z.literal('')),
})

export type AgentFormData = z.infer<typeof agentSchema>

export const agentExecuteSchema = z.object({
  prompt: z.string().min(1, 'Prompt is required').max(10000, 'Prompt must be under 10,000 characters'),
})

export type AgentExecuteFormData = z.infer<typeof agentExecuteSchema>

export const orchestrateSchema = z.object({
  goal: z.string().min(1, 'Goal is required').max(5000, 'Goal must be under 5,000 characters'),
  context: z.string().max(10000, 'Context must be under 10,000 characters').optional().or(z.literal('')),
})

export type OrchestrateFormData = z.infer<typeof orchestrateSchema>

export const settingsSchema = z.object({
  apiUrl: z.string().url('Must be a valid URL').optional().or(z.literal('')),
  hfToken: z.string().min(1, 'Token is required').optional().or(z.literal('')),
  customContext: z.string().max(10000, 'Instructions must be under 10,000 characters').optional().or(z.literal('')),
})

export type SettingsFormData = z.infer<typeof settingsSchema>

export const quickNoteSchema = z.object({
  text: z.string().min(1, 'Note is required').max(5000, 'Note must be under 5,000 characters'),
})

export type QuickNoteFormData = z.infer<typeof quickNoteSchema>

export const systemPromptSchema = z.object({
  prompt: z.string().min(1, 'System prompt is required').max(10000, 'Prompt must be under 10,000 characters'),
  presetName: z.string().min(1, 'Preset name is required').max(50, 'Name must be under 50 characters'),
})

export type SystemPromptFormData = z.infer<typeof systemPromptSchema>

export const snapshotSchema = z.object({
  name: z.string().min(1, 'Name is required').max(50, 'Name must be under 50 characters'),
})

export type SnapshotFormData = z.infer<typeof snapshotSchema>

export const renameSchema = z.object({
  name: z.string().min(1, 'Name is required').max(200, 'Name must be under 200 characters'),
})

export type RenameFormData = z.infer<typeof renameSchema>
