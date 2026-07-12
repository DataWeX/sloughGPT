import React from 'react';
import {render} from '../../test-utils';
import {MessageBubble} from '../MessageBubble';

jest.mock('../../services/clipboard', () => ({copyToClipboard: jest.fn()}));
jest.mock('../../services/haptics', () => ({triggerHaptic: jest.fn()}));
jest.mock('../../services/sounds', () => ({sounds: {play: jest.fn()}}));
jest.mock('../../services/bookmarks', () => ({addBookmark: jest.fn(), removeBookmark: jest.fn(), isBookmarked: jest.fn(() => Promise.resolve(false))}));
jest.mock('../../services/pins', () => ({pinMessage: jest.fn(), unpinMessage: jest.fn(), isPinned: jest.fn(() => Promise.resolve(false))}));
jest.mock('../../services/reactions', () => ({getMessageReactions: jest.fn(() => Promise.resolve({})), toggleReaction: jest.fn(), REACTION_EMOJIS: []}));
jest.mock('../../services/toast', () => ({toast: {success: jest.fn(), error: jest.fn()}}));
jest.mock('tamagui', () => {
  const actual = jest.requireActual('tamagui');
  return {...actual, useTheme: jest.fn(() => ({color: '#000', color10: '#999'}))};
});

const baseMessage = {
  id: 'm1',
  role: 'user' as const,
  content: 'Hello world',
  timestamp: Date.now(),
};

describe('MessageBubble', () => {
  it('renders user message', () => {
    expect(() => render(<MessageBubble message={baseMessage} />)).not.toThrow();
  });

  it('renders assistant message', () => {
    expect(() =>
      render(<MessageBubble message={{...baseMessage, id: 'm2', role: 'assistant', content: 'Hi there'}} />),
    ).not.toThrow();
  });

  it('renders message with images', () => {
    expect(() =>
      render(<MessageBubble message={{...baseMessage, images: ['data:image/png;base64,abc']}} />),
    ).not.toThrow();
  });
});
