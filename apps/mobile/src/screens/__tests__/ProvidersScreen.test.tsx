import React from 'react';
import {render} from '@/test-utils';
import {ProvidersScreen} from '../ProvidersScreen';

jest.mock('../../stores/providers-store');
jest.mock('../../stores/hybrid-inference-store', () => ({
  useHybridStore: jest.fn(() => ({route: 'self-hosted'})),
}));
jest.mock('../../services/haptics', () => ({triggerHaptic: jest.fn()}));
jest.mock('../../services/sounds', () => ({sounds: {send: jest.fn(() => Promise.resolve())}}));
jest.mock('../../services/toast', () => ({toast: {success: jest.fn(), error: jest.fn(), warn: jest.fn()}}));

beforeEach(() => {
  jest.clearAllMocks();
  const {useProvidersStore} = require('../../stores/providers-store');
  useProvidersStore.mockImplementation((selectorOrKey?: any) => {
    const state = {
      providers: {
        openai: {id: 'openai', name: 'OpenAI', apiKey: '', baseUrl: 'https://api.openai.com/v1', defaultModel: 'gpt-4o-mini', enabled: true},
        anthropic: {id: 'anthropic', name: 'Anthropic', apiKey: 'sk-ant-test', baseUrl: 'https://api.anthropic.com/v1', defaultModel: 'claude-3-5-sonnet-20241022', enabled: true},
        google: {id: 'google', name: 'Google Gemini', apiKey: '', baseUrl: 'https://generativelanguage.googleapis.com/v1beta', defaultModel: 'gemini-2.0-flash', enabled: true},
        mistral: {id: 'mistral', name: 'Mistral', apiKey: '', baseUrl: 'https://api.mistral.ai/v1', defaultModel: 'mistral-small-latest', enabled: true},
        groq: {id: 'groq', name: 'Groq', apiKey: '', baseUrl: 'https://api.groq.com/openai/v1', defaultModel: 'llama-3.1-8b-instant', enabled: true},
        together: {id: 'together', name: 'Together AI', apiKey: '', baseUrl: 'https://api.together.xyz/v1', defaultModel: 'meta-llama/Llama-3-8b-chat-hf', enabled: true},
        deepseek: {id: 'deepseek', name: 'DeepSeek', apiKey: '', baseUrl: 'https://api.deepseek.com/v1', defaultModel: 'deepseek-chat', enabled: true},
        openrouter: {id: 'openrouter', name: 'OpenRouter', apiKey: '', baseUrl: 'https://openrouter.ai/api/v1', defaultModel: 'openai/gpt-3.5-turbo', enabled: true},
        custom: {id: 'custom', name: 'Custom', apiKey: '', baseUrl: 'http://localhost:11434/v1', defaultModel: '', enabled: true},
      },
      activeProviderId: 'anthropic',
      setActiveProvider: jest.fn().mockResolvedValue(undefined),
      setApiKey: jest.fn().mockResolvedValue(undefined),
      setBaseUrl: jest.fn().mockResolvedValue(undefined),
      setDefaultModel: jest.fn().mockResolvedValue(undefined),
      toggleProvider: jest.fn().mockResolvedValue(undefined),
    };
    return typeof selectorOrKey === 'function' ? selectorOrKey(state) : state;
  });
});

describe('ProvidersScreen', () => {
  it('renders without throwing', () => {
    expect(() => render(<ProvidersScreen />)).not.toThrow();
  });
});
