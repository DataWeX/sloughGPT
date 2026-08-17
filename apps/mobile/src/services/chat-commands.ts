import {toast} from './toast';
import {api} from './api-client';

export interface ChatCommand {
  command: string;
  description: string;
  usage: string;
  execute(args: string[], context: CommandContext): Promise<CommandResult>;
}

export interface CommandContext {
  showToast: (msg: string, type?: 'success' | 'error' | 'info') => void;
  clearChat: () => void;
  setTemperature: (t: number) => void;
  setModel: (name: string) => Promise<void>;
  setSoul: (name: string) => Promise<void>;
  exportChat: () => void;
  attachFile: () => void;
  searchKnowledge: (query: string) => Promise<void>;
  navigateTo: (path: string) => void;
  addSystemMessage: (content: string) => void;
  sendMessage: (text?: string) => Promise<void>;
  archiveConversation: () => void;
  renameConversation: (name: string) => void;
  searchConversations: (query: string) => void;
}

export interface CommandResult {
  handled: boolean;
  message?: string;
}

const commands: ChatCommand[] = [
  {
    command: '/help',
    description: 'Show available commands',
    usage: '/help',
    async execute(_args, ctx) {
      const list = commands
        .filter(c => c.command !== '/help')
        .map(c => `  ${c.usage} — ${c.description}`)
        .join('\n');
      ctx.addSystemMessage(`**Chat Commands**\n\n${list}`);
      return {handled: true};
    },
  },
  {
    command: '/clear',
    description: 'Clear the chat history',
    usage: '/clear',
    async execute(_args, ctx) {
      ctx.clearChat();
      ctx.showToast('Chat cleared', 'success');
      return {handled: true};
    },
  },
  {
    command: '/temp',
    description: 'Set the temperature (0.0 – 2.0)',
    usage: '/temp <value>',
    async execute(args, ctx) {
      const val = parseFloat(args[0]);
      if (isNaN(val) || val < 0 || val > 2) {
        ctx.showToast('Usage: /temp <0.0–2.0>', 'error');
        return {handled: true, message: 'Temperature must be between 0.0 and 2.0'};
      }
      ctx.setTemperature(val);
      ctx.showToast(`Temperature set to ${val}`, 'success');
      return {handled: true};
    },
  },
  {
    command: '/model',
    description: 'Switch the active model',
    usage: '/model <name>',
    async execute(args, ctx) {
      if (!args.length) {
        ctx.showToast('Usage: /model <name>', 'error');
        return {handled: true};
      }
      const name = args.join(' ');
      ctx.addSystemMessage(`Loading model **${name}**...`);
      try {
        await ctx.setModel(name);
        ctx.addSystemMessage(`Switched to model **${name}**`);
      } catch {
        ctx.addSystemMessage(`Failed to load model **${name}**`);
      }
      return {handled: true};
    },
  },
  {
    command: '/soul',
    description: 'Switch the active soul/personality',
    usage: '/soul <name>',
    async execute(args, ctx) {
      if (!args.length) {
        ctx.showToast('Usage: /soul <name>', 'error');
        return {handled: true};
      }
      const name = args.join(' ');
      try {
        await ctx.setSoul(name);
        ctx.addSystemMessage(`Switched to soul **${name}**`);
      } catch {
        ctx.addSystemMessage(`Soul **${name}** not found`);
      }
      return {handled: true};
    },
  },
  {
    command: '/export',
    description: 'Export the current chat as markdown',
    usage: '/export',
    async execute(_args, ctx) {
      ctx.exportChat();
      ctx.showToast('Chat exported', 'success');
      return {handled: true};
    },
  },
  {
    command: '/file',
    description: 'Attach a file to read and ask about',
    usage: '/file',
    async execute(_args, ctx) {
      ctx.attachFile();
      return {handled: true};
    },
  },
  {
    command: '/knowledge',
    description: 'Search the knowledge base',
    usage: '/knowledge <query>',
    async execute(args, ctx) {
      if (!args.length) {
        ctx.showToast('Usage: /knowledge <query>', 'error');
        return {handled: true};
      }
      const query = args.join(' ');
      ctx.addSystemMessage(`Searching knowledge base for "${query}"...`);
      await ctx.searchKnowledge(query);
      return {handled: true};
    },
  },
  {
    command: '/goto',
    description: 'Navigate to a page',
    usage: '/goto <path>',
    async execute(args, ctx) {
      if (!args.length) {
        ctx.showToast('Usage: /goto <path>', 'error');
        return {handled: true};
      }
      ctx.navigateTo(args[0]);
      return {handled: true};
    },
  },
  {
    command: '/summarize',
    description: 'Ask the model to summarise the conversation',
    usage: '/summarize',
    async execute(_args, ctx) {
      ctx.addSystemMessage('Please summarise our conversation so far in 3–5 concise bullet points.');
      await ctx.sendMessage();
      return {handled: true};
    },
  },
  {
    command: '/search',
    description: 'Search past conversations',
    usage: '/search <query>',
    async execute(args, ctx) {
      if (!args.length) {
        ctx.showToast('Usage: /search <query>', 'error');
        return {handled: true};
      }
      const query = args.join(' ');
      ctx.searchConversations(query);
      return {handled: true};
    },
  },
  {
    command: '/rename',
    description: 'Rename the current conversation',
    usage: '/rename <name>',
    async execute(args, ctx) {
      if (!args.length) {
        ctx.showToast('Usage: /rename <name>', 'error');
        return {handled: true};
      }
      const name = args.join(' ');
      ctx.renameConversation(name);
      ctx.showToast(`Renamed to "${name}"`, 'success');
      return {handled: true};
    },
  },
];

export function getAllCommands(): ChatCommand[] {
  return [...commands];
}
