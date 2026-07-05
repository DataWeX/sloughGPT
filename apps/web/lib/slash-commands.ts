export interface SlashCommand {
  name: string
  description: string
  icon: string
  prompt: (text: string) => string
}

export const slashCommands: SlashCommand[] = [
  {
    name: 'summarize',
    description: 'Summarize the text concisely',
    icon: '📝',
    prompt: (text) => `Summarize the following text concisely:\n\n${text}`,
  },
  {
    name: 'translate',
    description: 'Translate to English',
    icon: '🌐',
    prompt: (text) => `Translate the following text to English:\n\n${text}`,
  },
  {
    name: 'explain',
    description: 'Explain in simple terms',
    icon: '💡',
    prompt: (text) => `Explain the following in simple, easy-to-understand terms:\n\n${text}`,
  },
  {
    name: 'code',
    description: 'Generate code from description',
    icon: '💻',
    prompt: (text) => `Generate clean, well-commented code for the following:\n\n${text}`,
  },
  {
    name: 'debug',
    description: 'Debug and fix code',
    icon: '🐛',
    prompt: (text) => `Debug the following code, identify issues, and provide a fixed version:\n\n${text}`,
  },
  {
    name: 'rewrite',
    description: 'Rewrite for clarity',
    icon: '✏️',
    prompt: (text) => `Rewrite the following text for better clarity and flow:\n\n${text}`,
  },
  {
    name: 'expand',
    description: 'Expand with more detail',
    icon: '📖',
    prompt: (text) => `Expand the following text with more detail and examples:\n\n${text}`,
  },
  {
    name: 'bullet',
    description: 'Convert to bullet points',
    icon: '•',
    prompt: (text) => `Convert the following text into clear, concise bullet points:\n\n${text}`,
  },
  {
    name: 'formal',
    description: 'Make more formal',
    icon: '🎩',
    prompt: (text) => `Rewrite the following text in a more formal, professional tone:\n\n${text}`,
  },
  {
    name: 'casual',
    description: 'Make more casual',
    icon: '😊',
    prompt: (text) => `Rewrite the following text in a more casual, friendly tone:\n\n${text}`,
  },
]

export function findMatchingCommands(query: string): SlashCommand[] {
  const search = query.toLowerCase().trim()
  if (!search) return slashCommands
  return slashCommands.filter(
    (cmd) =>
      cmd.name.includes(search) ||
      cmd.description.toLowerCase().includes(search)
  )
}

export function parseSlashCommand(input: string): { command: SlashCommand | null; text: string } {
  const match = input.match(/^\/(\w+)\s*([\s\S]*)$/)
  if (!match) return { command: null, text: input }

  const [, cmdName, text] = match
  const command = slashCommands.find((c) => c.name === cmdName)
  return { command: command || null, text: text.trim() }
}
