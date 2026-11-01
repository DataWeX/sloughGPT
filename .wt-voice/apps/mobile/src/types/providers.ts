/**
 * Third-party inference provider types.
 *
 * Each provider has a unique API shape for chat completions.
 * Providers are OpenAI-compatible by default (OpenAI, Groq, Together, etc.)
 * with special handling for Anthropic and Google.
 */

/** Supported provider IDs. */
export type ProviderId = 'openai' | 'anthropic' | 'google' | 'mistral' | 'groq' | 'together' | 'deepseek' | 'openrouter' | 'custom';

/** Inference engine — extends existing ActiveEngine with third-party providers. */
export type InferenceTarget = 'slonet' | 'qwen' | 'remote' | ProviderId;

/** A message in the chat conversation. */
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

/** Configuration for a single provider. */
export interface ProviderConfig {
  id: ProviderId;
  name: string;
  baseUrl: string;
  apiKey: string;
  /** Default model to use for this provider. */
  defaultModel: string;
  /** Whether this provider is enabled. */
  enabled: boolean;
  /** Custom headers (e.g., for OpenRouter). */
  headers?: Record<string, string>;
}

/** Runtime state for a provider (not persisted). */
export interface ProviderState {
  /** Whether a request is currently in flight. */
  loading: boolean;
  /** Last error message, if any. */
  error: string | null;
}

/** Standard chat completion request body (OpenAI format). */
export interface OpenAIRequestBody {
  model: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream: boolean;
}

/** Anthropic-specific request body. */
export interface AnthropicRequestBody {
  model: string;
  messages: ChatMessage[];
  max_tokens: number;
  temperature?: number;
  top_p?: number;
  system?: string;
  stream: boolean;
}

/** Google Gemini request body. */
export interface GeminiRequestBody {
  contents: Array<{
    role: 'user' | 'model';
    parts: Array<{text: string}>;
  }>;
  generationConfig: {
    maxOutputTokens: number;
    temperature?: number;
    topP?: number;
    topK?: number;
  };
  systemInstruction?: {
    parts: Array<{text: string}>;
  };
}

/** Provider model catalog entry. */
export interface ProviderModel {
  id: string;
  name: string;
  provider: ProviderId;
  contextWindow?: number;
  maxOutput?: number;
}

/** Stream callback types. */
export type OnTokenCallback = (token: string) => void;
export type OnCompleteCallback = (fullText: string) => void;
export type OnErrorCallback = (error: string) => void;

/** Stream result for non-streaming calls. */
export interface StreamResult {
  text: string;
  model: string;
  tokensUsed?: number;
  finishReason?: string;
}

/** Built-in provider registry — default configs for known providers. */
export const PROVIDER_REGISTRY: Record<ProviderId, Omit<ProviderConfig, 'apiKey'>> = {
  openai: {
    id: 'openai',
    name: 'OpenAI',
    baseUrl: 'https://api.openai.com/v1',
    defaultModel: 'gpt-4o-mini',
    enabled: true,
  },
  anthropic: {
    id: 'anthropic',
    name: 'Anthropic',
    baseUrl: 'https://api.anthropic.com/v1',
    defaultModel: 'claude-3-5-sonnet-20241022',
    enabled: true,
  },
  google: {
    id: 'google',
    name: 'Google Gemini',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta',
    defaultModel: 'gemini-2.0-flash',
    enabled: true,
  },
  mistral: {
    id: 'mistral',
    name: 'Mistral AI',
    baseUrl: 'https://api.mistral.ai/v1',
    defaultModel: 'mistral-small-latest',
    enabled: true,
  },
  groq: {
    id: 'groq',
    name: 'Groq',
    baseUrl: 'https://api.groq.com/openai/v1',
    defaultModel: 'llama-3.1-8b-instant',
    enabled: true,
  },
  together: {
    id: 'together',
    name: 'Together AI',
    baseUrl: 'https://api.together.xyz/v1',
    defaultModel: 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    enabled: true,
  },
  deepseek: {
    id: 'deepseek',
    name: 'DeepSeek',
    baseUrl: 'https://api.deepseek.com/v1',
    defaultModel: 'deepseek-chat',
    enabled: true,
  },
  openrouter: {
    id: 'openrouter',
    name: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    defaultModel: 'openai/gpt-4o-mini',
    enabled: true,
  },
  custom: {
    id: 'custom',
    name: 'Custom',
    baseUrl: 'http://localhost:11434/v1',
    defaultModel: 'model',
    enabled: true,
  },
};

/** Models per provider (curated list — user can always type a custom model ID). */
export const PROVIDER_MODELS: Record<ProviderId, ProviderModel[]> = {
  openai: [
    {id: 'gpt-4o', name: 'GPT-4o', provider: 'openai', contextWindow: 128000, maxOutput: 16384},
    {id: 'gpt-4o-mini', name: 'GPT-4o Mini', provider: 'openai', contextWindow: 128000, maxOutput: 16384},
    {id: 'gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'openai', contextWindow: 128000, maxOutput: 4096},
    {id: 'o1-mini', name: 'o1 Mini', provider: 'openai', contextWindow: 128000, maxOutput: 65536},
  ],
  anthropic: [
    {id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4', provider: 'anthropic', contextWindow: 200000, maxOutput: 64000},
    {id: 'claude-3-5-sonnet-20241022', name: 'Claude 3.5 Sonnet', provider: 'anthropic', contextWindow: 200000, maxOutput: 8192},
    {id: 'claude-3-5-haiku-20241022', name: 'Claude 3.5 Haiku', provider: 'anthropic', contextWindow: 200000, maxOutput: 8192},
  ],
  google: [
    {id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash', provider: 'google', contextWindow: 1048576, maxOutput: 8192},
    {id: 'gemini-2.5-pro-preview-05-06', name: 'Gemini 2.5 Pro', provider: 'google', contextWindow: 1048576, maxOutput: 65536},
    {id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash', provider: 'google', contextWindow: 1048576, maxOutput: 8192},
  ],
  mistral: [
    {id: 'mistral-small-latest', name: 'Mistral Small', provider: 'mistral', contextWindow: 32000, maxOutput: 8192},
    {id: 'mistral-medium-latest', name: 'Mistral Medium', provider: 'mistral', contextWindow: 32000, maxOutput: 8192},
    {id: 'codestral-latest', name: 'Codestral', provider: 'mistral', contextWindow: 32000, maxOutput: 8192},
  ],
  groq: [
    {id: 'llama-3.1-8b-instant', name: 'Llama 3.1 8B', provider: 'groq', contextWindow: 131072, maxOutput: 8192},
    {id: 'llama-3.3-70b-versatile', name: 'Llama 3.3 70B', provider: 'groq', contextWindow: 131072, maxOutput: 32768},
    {id: 'mixtral-8x7b-32768', name: 'Mixtral 8x7B', provider: 'groq', contextWindow: 32768, maxOutput: 32768},
  ],
  together: [
    {id: 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', name: 'Llama 3.1 8B Turbo', provider: 'together', contextWindow: 131072},
    {id: 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', name: 'Llama 3.1 70B Turbo', provider: 'together', contextWindow: 131072},
    {id: 'Qwen/Qwen2.5-72B-Instruct-Turbo', name: 'Qwen 2.5 72B', provider: 'together', contextWindow: 32768},
  ],
  deepseek: [
    {id: 'deepseek-chat', name: 'DeepSeek Chat', provider: 'deepseek', contextWindow: 65536, maxOutput: 8192},
    {id: 'deepseek-coder', name: 'DeepSeek Coder', provider: 'deepseek', contextWindow: 65536, maxOutput: 8192},
  ],
  openrouter: [
    {id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini (via OpenRouter)', provider: 'openrouter', contextWindow: 128000},
    {id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet (via OpenRouter)', provider: 'openrouter', contextWindow: 200000},
    {id: 'meta-llama/llama-3.1-8b-instruct', name: 'Llama 3.1 8B (via OpenRouter)', provider: 'openrouter', contextWindow: 131072},
  ],
  custom: [],
};
