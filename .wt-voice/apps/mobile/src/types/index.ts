export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  images?: string[];
  /** Base64 encoded audio data or local file URI for voice messages. */
  audio?: string;
  /** Server-side audio file path (returned by voice message endpoint). */
  audio_path?: string;
  /** Duration of voice message in milliseconds. */
  audio_duration_ms?: number;
  /** True if this is a voice message (recorded audio, not text). */
  _voice?: boolean;
  /** Delivery status */
  status?: 'sending' | 'sent' | 'failed';
}

export interface Session {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  archived?: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  type: string;
  loaded: boolean;
  size_gb: number;
  size_mb: number;
  params: string;
  description: string;
  source: string;
  tags: string[];
  thumbnail?: string;
}

export interface SoulInfo {
  name: string;
  description: string;
  traits: string[];
}

export interface CheckpointInfo {
  name: string;
  soul: string;
  loss: number;
  steps: number;
  traits: Record<string, number>;
  created_at: string;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_name: string | null;
  model_type: string | null;
  uptime: number;
  inference_count: number;
}

export interface KnowledgeItem {
  id: string;
  content: string;
  topic: string | null;
  importance: number;
  created_at: string;
}

export interface DetailedHealth {
  api: {status: string; model_loaded: boolean; model_name: string | null};
  system: {
    cpu_percent: number;
    memory_percent: number;
    memory_used_gb: number;
    memory_total_gb: number;
    disk_used_gb: number;
    disk_free_gb: number;
    disk_total_gb: number;
    uptime: number;
  };
  inference?: {
    inference_count: number;
    avg_tokens_per_sec: number;
    total_tokens: number;
  };
  services?: {
    training_pool: {active: number; max: number; tracked: number};
    inference_pool: {workers: number; active: number; queue_timeout: number};
  };
}

export type ThemeMode = 'light' | 'dark' | 'system';

export interface SearchMatch {
  role: string;
  content: string;
  timestamp: string;
}

export interface SearchResult {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  match_count: number;
  matches: SearchMatch[];
}

export interface Adapter {
  id: string;
  user_id: string;
  name: string;
  loss: number | null;
  steps: number | null;
  traits: Record<string, number>;
  created_at: string;
}

export interface BenchmarkResult {
  model?: string;
  model_id?: string;
  timestamp?: string;
  coherence: number;
  repetition: number;
  perplexity: number | null;
  avg_length?: number;
  avg_response_length?: number;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  row_count: number;
  total_chars: number;
  format: string;
  source: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}
