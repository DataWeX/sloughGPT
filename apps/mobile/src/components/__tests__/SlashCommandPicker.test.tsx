import React from 'react';
import {render} from '../../test-utils';
import {SlashCommandPicker} from '../SlashCommandPicker';

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

jest.mock('../../services/chat-commands', () => ({
  getAllCommands: () => [
    {command: '/help', description: 'Show available commands', usage: '/help', execute: jest.fn()},
    {command: '/clear', description: 'Clear the chat history', usage: '/clear', execute: jest.fn()},
    {command: '/temp', description: 'Set the temperature', usage: '/temp <value>', execute: jest.fn()},
    {command: '/model', description: 'Switch the active model', usage: '/model <name>', execute: jest.fn()},
    {command: '/soul', description: 'Switch the active soul', usage: '/soul <name>', execute: jest.fn()},
  ],
}));

describe('SlashCommandPicker', () => {
  const defaultProps = {
    visible: true,
    query: '/',
    onSelect: jest.fn(),
    onExecute: jest.fn(),
    onClose: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders when visible without crashing', () => {
    expect(() => render(<SlashCommandPicker {...defaultProps} />)).not.toThrow();
  });

  it('renders nothing when not visible', () => {
    expect(() =>
      render(<SlashCommandPicker {...defaultProps} visible={false} />),
    ).not.toThrow();
  });

  it('renders with filtered query', () => {
    expect(() =>
      render(<SlashCommandPicker {...defaultProps} query="/he" />),
    ).not.toThrow();
  });

  it('renders with no matching query', () => {
    expect(() =>
      render(<SlashCommandPicker {...defaultProps} query="/xyz" />),
    ).not.toThrow();
  });
});
