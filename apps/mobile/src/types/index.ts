export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

export interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
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
}

export type ThemeMode = 'light' | 'dark' | 'system';
