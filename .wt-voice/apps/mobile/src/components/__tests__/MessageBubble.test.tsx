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
jest.mock('../../theme/colors', () => ({
  useColors: jest.fn(() => ({
    background: '#FFFFFF',
    backgroundHover: '#F4F2F8',
    border: '#E4E0F2',
    text: '#1A1625',
    textMuted: '#827A96',
    textSecondary: '#6B7280',
    textOnPrimary: '#FFFFFF',
    primary: '#7C52C4',
    primaryLight: '#B794F4',
    error: '#EF4444',
    errorLight: '#FDE8E8',
    success: '#22C55E',
    warning: '#F59E0B',
    white: '#FFFFFF',
    overlay: (o: number) => `rgba(0,0,0,${o})`,
    primaryAlpha: (o: number) => `rgba(124,82,196,${o})`,
    errorAlpha: (o: number) => `rgba(239,68,68,${o})`,
    errorDarkAlpha: (o: number) => `rgba(220,38,38,${o})`,
    successAlpha: (o: number) => `rgba(34,197,94,${o})`,
    whiteAlpha: (o: number) => `rgba(255,255,255,${o})`,
  })),
}));

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
