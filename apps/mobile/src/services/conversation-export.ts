/**
 * Conversation export — formats messages as markdown and shares via system share sheet.
 * Supports exporting a single conversation or all conversations.
 */

import {Share, Platform} from 'react-native';
import type {Message, Session} from '../types';

/**
 * Format a single message as markdown.
 */
function formatMessage(msg: Message): string {
  const role = msg.role === 'user' ? '**You**' : msg.role === 'assistant' ? '**Assistant**' : '**System**';
  const time = new Date(msg.timestamp).toLocaleString();
  const parts: string[] = [];

  parts.push(`### ${role}`);
  parts.push(`*${time}*`);
  parts.push('');

  if (msg.images && msg.images.length > 0) {
    parts.push(`*[${msg.images.length} image${msg.images.length > 1 ? 's' : ''}]*`);
    parts.push('');
  }

  if (msg._voice || msg.audio_path) {
    parts.push('*[voice message]*');
    parts.push('');
  }

  parts.push(msg.content);
  parts.push('');

  return parts.join('\n');
}

/**
 * Export a conversation as a formatted markdown string.
 */
export function exportConversationAsMarkdown(
  messages: Message[],
  sessionName?: string,
): string {
  const header: string[] = [];
  header.push(`# ${sessionName || 'Conversation'}`);
  header.push('');
  header.push(`*Exported on ${new Date().toLocaleString()}*`);
  header.push(`*${messages.length} messages*`);
  header.push('');
  header.push('---');
  header.push('');

  const body = messages.map(formatMessage).join('\n---\n\n');

  return header.join('\n') + body;
}

/**
 * Share a conversation via the system share sheet.
 * Returns true if the user completed the share action.
 */
export async function shareConversation(
  messages: Message[],
  sessionName?: string,
): Promise<boolean> {
  if (messages.length === 0) return false;

  const markdown = exportConversationAsMarkdown(messages, sessionName);
  const filename = `${(sessionName || 'conversation').replace(/[^a-zA-Z0-9]/g, '_')}.md`;

  try {
    const result = await Share.share(
      {
        title: sessionName || 'Conversation Export',
        message: markdown,
        // On iOS, share as file; on Android, share as text
        ...(Platform.OS === 'ios' ? {url: undefined} : {}),
      },
      {
        // Android: custom share dialog title
        dialogTitle: sessionName || 'Export Conversation',
      },
    );
    return result.action === Share.sharedAction;
  } catch {
    return false;
  }
}

/**
 * Export all sessions as a combined markdown document.
 */
export async function shareAllConversations(
  sessions: Session[],
  getMessages: (sessionId: string) => Message[],
): Promise<boolean> {
  if (sessions.length === 0) return false;

  const parts: string[] = [];
  parts.push('# SloughGPT Conversations');
  parts.push('');
  parts.push(`*Exported on ${new Date().toLocaleString()}*`);
  parts.push(`*${sessions.length} conversations*`);
  parts.push('');

  for (const session of sessions) {
    const messages = getMessages(session.id);
    if (messages.length === 0) continue;

    parts.push(`## ${session.name}`);
    parts.push('');
    parts.push(`*${messages.length} messages, created ${new Date(session.created_at).toLocaleString()}*`);
    parts.push('');
    parts.push('---');
    parts.push('');

    for (const msg of messages) {
      parts.push(formatMessage(msg));
      parts.push('---');
      parts.push('');
    }
  }

  if (parts.length <= 5) return false;

  const markdown = parts.join('\n');

  try {
    const result = await Share.share(
      {
        title: 'SloughGPT Conversations',
        message: markdown,
      },
      {dialogTitle: 'Export All Conversations'},
    );
    return result.action === Share.sharedAction;
  } catch {
    return false;
  }
}
