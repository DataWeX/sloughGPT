import {exportConversationAsMarkdown, shareConversation} from '../conversation-export';
import type {Message} from '../../types';

jest.mock('react-native', () => {
  const mockShare = jest.fn().mockResolvedValue({action: 'sharedAction'});
  return {
    Share: {
      share: mockShare,
      sharedAction: 'sharedAction',
      dismissedAction: 'dismissed',
    },
    Platform: {OS: 'ios'},
    __mockShare: mockShare,
  };
});

const {__mockShare: mockShare} = require('react-native');

function makeMsg(overrides: Partial<Message> = {}): Message {
  return {
    id: '1',
    role: 'user',
    content: 'Hello world',
    timestamp: 1700000000000,
    ...overrides,
  };
}

beforeEach(() => {
  jest.clearAllMocks();
});

describe('exportConversationAsMarkdown', () => {
  it('produces a markdown string with header', () => {
    const md = exportConversationAsMarkdown([makeMsg()], 'Test Chat');
    expect(md).toContain('# Test Chat');
    expect(md).toContain('1 messages');
  });

  it('uses default title when none provided', () => {
    const md = exportConversationAsMarkdown([makeMsg()]);
    expect(md).toContain('# Conversation');
  });

  it('formats user messages with "You" label', () => {
    const md = exportConversationAsMarkdown([makeMsg({role: 'user'})]);
    expect(md).toContain('**You**');
  });

  it('formats assistant messages with "Assistant" label', () => {
    const md = exportConversationAsMarkdown([makeMsg({role: 'assistant'})]);
    expect(md).toContain('**Assistant**');
  });

  it('formats system messages with "System" label', () => {
    const md = exportConversationAsMarkdown([makeMsg({role: 'system'})]);
    expect(md).toContain('**System**');
  });

  it('includes message content', () => {
    const md = exportConversationAsMarkdown([makeMsg({content: 'test content 123'})]);
    expect(md).toContain('test content 123');
  });

  it('marks images as [N images]', () => {
    const md = exportConversationAsMarkdown([makeMsg({images: ['a', 'b']})]);
    expect(md).toContain('[2 images]');
  });

  it('marks voice messages', () => {
    const md = exportConversationAsMarkdown([makeMsg({_voice: true})]);
    expect(md).toContain('[voice message]');
  });

  it('includes message count in header', () => {
    const msgs = [makeMsg({id: '1'}), makeMsg({id: '2'}), makeMsg({id: '3'})];
    const md = exportConversationAsMarkdown(msgs);
    expect(md).toContain('3 messages');
  });

  it('separates messages with horizontal rules', () => {
    const msgs = [makeMsg({id: '1'}), makeMsg({id: '2'})];
    const md = exportConversationAsMarkdown(msgs);
    expect(md).toContain('---');
  });

  it('returns empty header for no messages', () => {
    const md = exportConversationAsMarkdown([]);
    expect(md).toContain('# Conversation');
    expect(md).toContain('0 messages');
  });
});

describe('shareConversation', () => {
  it('calls Share.share with formatted markdown', async () => {
    const msgs = [makeMsg({content: 'hi'})];
    const result = await shareConversation(msgs, 'My Chat');
    expect(mockShare).toHaveBeenCalledTimes(1);
    const call = mockShare.mock.calls[0][0];
    expect(call.title).toBe('My Chat');
    expect(call.message).toContain('hi');
    expect(result).toBe(true);
  });

  it('returns false for empty messages', async () => {
    const result = await shareConversation([]);
    expect(mockShare).not.toHaveBeenCalled();
    expect(result).toBe(false);
  });

  it('returns false when share is dismissed', async () => {
    mockShare.mockResolvedValueOnce({action: 'dismissed'});
    const result = await shareConversation([makeMsg()]);
    expect(result).toBe(false);
  });

  it('returns false on share error', async () => {
    mockShare.mockRejectedValueOnce(new Error('share failed'));
    const result = await shareConversation([makeMsg()]);
    expect(result).toBe(false);
  });
});
