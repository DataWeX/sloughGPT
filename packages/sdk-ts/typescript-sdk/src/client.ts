export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface GenerateRequest {
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  personality?: string;
  model?: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  model?: string;
  temperature?: number;
  max_new_tokens?: number;
  top_p?: number;
  top_k?: number;
}

export interface GenerationResult {
  text: string;
  model: string;
  personality?: string;
  inference_time_ms?: number;
}

export interface ChatResult {
  message: ChatMessage;
  model: string;
  inference_time_ms?: number;
  tokens_generated?: number;
}

export interface HealthStatus {
  status: 'healthy' | 'unhealthy';
  model_loaded: boolean;
  model_type: string;
}

export interface SystemInfo {
  name: string;
  version: string;
  model: { type: string; loaded: boolean };
}

export interface ModelInfo {
  model_id: string;
  name: string;
  description?: string;
  model_type?: string;
}

export interface MetricsData {
  requests_today: number;
  tokens_today: number;
  cache_hit_rate: number;
}

export interface TrainingJob {
  id: string
  name?: string
  model?: string
  dataset?: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  epochs?: number
  current_epoch?: number
  global_step?: number
  loss?: number
  train_loss?: number
  eval_loss?: number
  data_path?: string
  output_checkpoint_stem?: string
  data_source?: string
  checkpoint?: string
  error?: string
}

export interface TrainingStartPayload {
  name: string
  model: string
  dataset?: string
  manifest_uri?: string
  dataset_ref?: { dataset_id: string; version: string; manifest_uri: string }
  epochs?: number
  batch_size?: number
  learning_rate?: number
  n_embed?: number
  n_layer?: number
  n_head?: number
  block_size?: number
  max_steps?: number
  log_interval?: number
  eval_interval?: number
}

export interface Experiment {
  experiment_id: string;
  name: string;
  description?: string;
  metrics?: Record<string, number>;
}

export interface SoulProfile {
  name: string;
  description: string;
  traits?: Record<string, number>;
}

export interface KnowledgeItem {
  id: string;
  content: string;
  topic?: string;
  source?: string;
  importance?: number;
}

export interface KnowledgSearchResult {
  id: string;
  content: string;
  relevance: number;
}

export interface TokenizerStats {
  vocab_size: number;
  num_tokens: number;
  num_merges?: number;
}

export interface SystemMetrics {
  cpu_percent: number;
  memory_percent: number;
  disk_percent: number;
  uptime_seconds: number;
  gpu_available: boolean;
  gpu_percent?: number;
}

export interface WorkflowStatus {
  status: string;
  active: boolean;
  feedback_count: number;
  last_aggregation?: string;
}

export interface FeedbackRecord {
  session_id: string;
  message_id: string;
  score: number;
  tags?: string[];
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface SloughGPTConfig {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  headers?: Record<string, string>;
  onLog?: (level: LogLevel, message: string) => void;
}

export class SloughGPTError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public response?: unknown
  ) {
    super(message);
    this.name = 'SloughGPTError';
  }
}

export class SloughGPTClient {
  private baseUrl: string;
  private timeout: number;
  private headers: Record<string, string>;
  private onLog?: (level: LogLevel, message: string) => void;

  constructor(config: SloughGPTConfig = {}) {
    this.baseUrl = (config.baseUrl || 'http://localhost:8000').replace(/\/$/, '');
    this.timeout = config.timeout || 30000;
    this.headers = {
      'Content-Type': 'application/json',
      ...(config.apiKey ? { 'X-API-Key': config.apiKey } : {}),
      ...config.headers,
    };
    this.onLog = config.onLog;
  }

  private log(level: LogLevel, message: string) {
    if (this.onLog) {
      this.onLog(level, message);
    }
  }

  private async request<T>(
    method: string,
    endpoint: string,
    body?: unknown
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    this.log('debug', `${method} ${url}`);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers: this.headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new SloughGPTError(
          `HTTP ${response.status}: ${response.statusText}`,
          response.status
        );
      }

      return await response.json();
    } catch (error: unknown) {
      clearTimeout(timeoutId);
      if ((error as { name?: string }).name === 'AbortError') {
        throw new SloughGPTError('Request timeout', 408);
      }
      throw error;
    }
  }

  // ============ Health & Info ============

  async health(): Promise<HealthStatus> {
    return this.request<HealthStatus>('GET', '/health');
  }

  async liveness(): Promise<{ status: string }> {
    return this.request('GET', '/health/live');
  }

  async readiness(): Promise<{ status: string; model_loaded: boolean }> {
    return this.request('GET', '/health/ready');
  }

  async detailedHealth(): Promise<Record<string, unknown>> {
    return this.request('GET', '/health/detailed');
  }

  async info(): Promise<SystemInfo> {
    return this.request<SystemInfo>('GET', '/info');
  }

  // ============ Inference & Generation ============

  async generate(request: GenerateRequest): Promise<GenerationResult> {
    this.log('info', `Generating: "${request.prompt.slice(0, 50)}..."`);
    return this.request<GenerationResult>('POST', '/inference/generate', {
      prompt: request.prompt,
      max_new_tokens: request.max_new_tokens || 100,
      temperature: request.temperature || 0.8,
      top_k: request.top_k || 50,
      top_p: request.top_p || 0.9,
      personality: request.personality,
      model: request.model,
    });
  }

  async *generateStream(
    request: GenerateRequest
  ): AsyncGenerator<string, void, unknown> {
    const url = `${this.baseUrl}/inference/generate/stream`;
    this.log('info', `Streaming: "${request.prompt.slice(0, 50)}..."`);

    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        prompt: request.prompt,
        max_new_tokens: request.max_new_tokens || 100,
        temperature: request.temperature || 0.8,
        personality: request.personality,
        model: request.model,
      }),
    });

    if (!response.ok) {
      throw new SloughGPTError(`HTTP ${response.status}`, response.status);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new SloughGPTError('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data:')) {
          const data = line.slice(5).trim();
          if (data && data !== '[DONE]') {
            yield data;
          }
        }
      }
    }
  }

  async chat(request: ChatRequest): Promise<ChatResult> {
    this.log('info', `Chat: ${request.messages.length} messages`);
    const raw = await this.request<{
      text?: string;
      model?: string;
      tokens_generated?: number;
      error?: string;
    }>('POST', '/chat', {
      messages: request.messages,
      model: request.model,
      temperature: request.temperature ?? 0.8,
      max_new_tokens: request.max_new_tokens ?? 100,
      top_p: request.top_p ?? 0.9,
      top_k: request.top_k ?? 50,
    });

    const content = raw.text ?? '';
    if (raw.error && !content) {
      throw new SloughGPTError(raw.error, 400);
    }

    return {
      message: { role: 'assistant', content },
      model: raw.model ?? 'unknown',
      tokens_generated: raw.tokens_generated,
    };
  }

  async *chatStream(
    request: ChatRequest
  ): AsyncGenerator<string, void, unknown> {
    const url = `${this.baseUrl}/chat/stream`;
    this.log('info', `Chat stream: ${request.messages.length} messages`);

    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        messages: request.messages,
        model: request.model,
        temperature: request.temperature ?? 0.8,
        max_new_tokens: request.max_new_tokens ?? 100,
        top_p: request.top_p ?? 0.9,
        top_k: request.top_k ?? 50,
      }),
    });

    if (!response.ok) {
      throw new SloughGPTError(`HTTP ${response.status}`, response.status);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new SloughGPTError('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trimEnd();
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload || payload === '[DONE]') continue;
        try {
          const envelope = JSON.parse(payload) as {
            stream?: string;
            phase?: string;
            status?: string;
            data?: { token?: string; error?: string };
            error?: string;
            message?: string;
          };
          if (envelope.status === 'error') {
            throw new SloughGPTError(envelope.data?.error || envelope.message || 'Stream error', 500);
          }
          if (envelope.status === 'complete') {
            return;
          }
          if (envelope.data?.token) {
            yield envelope.data.token;
          }
        } catch (e) {
          if (e instanceof SloughGPTError) {
            throw e;
          }
        }
      }
    }
  }

  // ============ Models ============

  async listModels(): Promise<ModelInfo[]> {
    return this.request<ModelInfo[]>('GET', '/models');
  }

  async loadModel(modelId: string): Promise<{ status: string }> {
    return this.request('POST', '/models/load', { model_id: modelId });
  }

  async unloadModel(): Promise<{ status: string }> {
    return this.request('POST', '/models/unload');
  }

  async getCurrentModel(): Promise<Record<string, unknown>> {
    return this.request('GET', '/models/current');
  }

  async listHuggingFaceModels(query?: string, limit?: number): Promise<Record<string, unknown>[]> {
    const params = new URLSearchParams();
    if (query) params.set('q', query);
    if (limit) params.set('limit', String(limit));
    const qs = params.toString();
    return this.request<Record<string, unknown>[]>('GET', `/models/hf${qs ? '?' + qs : ''}`);
  }

  // ============ Sessions ============

  async createSession(): Promise<Record<string, unknown>> {
    return this.request('POST', '/chat/sessions');
  }

  async listSessions(): Promise<Record<string, unknown>[]> {
    return this.request<Record<string, unknown>[]>('GET', '/chat/sessions');
  }

  async getSession(sessionId: string): Promise<Record<string, unknown>> {
    return this.request('GET', `/chat/sessions/${sessionId}`);
  }

  async deleteSession(sessionId: string): Promise<Record<string, unknown>> {
    return this.request('DELETE', `/chat/sessions/${sessionId}`);
  }

  async saveSessionContext(sessionId: string, context: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.request('POST', `/session/${sessionId}/context`, context);
  }

  async getSessionMessages(sessionId: string): Promise<Record<string, unknown>[]> {
    return this.request<Record<string, unknown>[]>('GET', `/session/${sessionId}/messages`);
  }

  async *regenerateStream(sessionId: string): AsyncGenerator<string, void, unknown> {
    const url = `${this.baseUrl}/session/${sessionId}/regenerate`;
    const response = await fetch(url, { method: 'POST', headers: this.headers });
    if (!response.ok) throw new SloughGPTError(`HTTP ${response.status}`, response.status);

    const reader = response.body?.getReader();
    if (!reader) throw new SloughGPTError('No response body');
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const trimmed = line.trimEnd();
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload || payload === '[DONE]') continue;
        try {
          const envelope = JSON.parse(payload) as { data?: { token?: string }; status?: string };
          if (envelope.status === 'complete') return;
          if (envelope.data?.token) yield envelope.data.token;
        } catch { /* skip */ }
      }
    }
  }

  // ============ Souls ============

  async listSouls(): Promise<SoulProfile[]> {
    return this.request<SoulProfile[]>('GET', '/souls');
  }

  async getCurrentSoul(): Promise<SoulProfile> {
    return this.request<SoulProfile>('GET', '/souls/current');
  }

  async switchSoul(name: string, checkpointName?: string): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {};
    if (checkpointName) body.checkpoint_name = checkpointName;
    return this.request('POST', `/souls/switch/${name}`, body);
  }

  // ============ Knowledge ============

  async listKnowledge(): Promise<KnowledgeItem[]> {
    return this.request<KnowledgeItem[]>('GET', '/knowledge');
  }

  async addKnowledge(content: string, topic?: string): Promise<KnowledgeItem> {
    return this.request<KnowledgeItem>('POST', '/knowledge', { content, topic });
  }

  async deleteKnowledge(itemId: string): Promise<Record<string, unknown>> {
    return this.request('DELETE', `/knowledge/${itemId}`);
  }

  async searchKnowledge(query: string): Promise<KnowledgSearchResult[]> {
    return this.request<KnowledgSearchResult[]>('GET', `/knowledge/search?q=${encodeURIComponent(query)}`);
  }

  async getKnowledgeStats(): Promise<Record<string, unknown>> {
    return this.request('GET', '/knowledge/stats');
  }

  async getKnowledgeTopics(): Promise<string[]> {
    return this.request<string[]>('GET', '/knowledge/topics');
  }

  async ingestKnowledgeUrl(url: string): Promise<Record<string, unknown>> {
    return this.request('POST', '/knowledge/ingest-url', { url });
  }

  // ============ Tokenizer ============

  async getTokenizerStats(): Promise<TokenizerStats> {
    return this.request<TokenizerStats>('GET', '/tokenizer/stats');
  }

  async tokenize(text: string): Promise<{ tokens: number[]; token_count: number }> {
    return this.request('POST', '/tokenizer/tokenize', { text });
  }

  async trainTokenizer(text: string, vocabSize?: number): Promise<Record<string, unknown>> {
    return this.request('POST', '/tokenizer/train', { text, vocab_size: vocabSize });
  }

  // ============ System ============

  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.request<SystemMetrics>('GET', '/system/metrics');
  }

  async getSystemInfo(): Promise<Record<string, unknown>> {
    return this.request('GET', '/system/info');
  }

  async getSystemDisk(): Promise<Record<string, unknown>> {
    return this.request('GET', '/system/disk');
  }

  // ============ Companion / Personality ============

  async getPersonalities(): Promise<Array<{ name: string; description: string }>> {
    return this.request('GET', '/personalities');
  }

  async setPersonality(personality: string): Promise<{ status: string }> {
    return this.request('POST', '/companion/personality', { personality });
  }

  async getCompanionPrompt(): Promise<{ prompt: string }> {
    return this.request('GET', '/companion/prompt');
  }

  async listCompanionPresets(): Promise<Record<string, unknown>[]> {
    return this.request<Record<string, unknown>[]>('GET', '/companion/presets');
  }

  // ============ Training ============

  async startTraining(input: TrainingStartPayload): Promise<TrainingJob> {
    return this.request<TrainingJob>('POST', '/training/start', input);
  }

  async getTrainingStatus(jobId: string): Promise<TrainingJob> {
    return this.request<TrainingJob>('GET', `/training/jobs/${jobId}`);
  }

  async listTrainingJobs(): Promise<TrainingJob[]> {
    return this.request<TrainingJob[]>('GET', '/training/jobs');
  }

  async deleteTrainingJob(jobId: string): Promise<Record<string, unknown>> {
    return this.request('DELETE', `/training/jobs/${jobId}`);
  }

  async stopTraining(): Promise<Record<string, unknown>> {
    return this.request('POST', '/training/control/stop');
  }

  async pauseTraining(): Promise<Record<string, unknown>> {
    return this.request('POST', '/training/control/pause');
  }

  async resumeTraining(): Promise<Record<string, unknown>> {
    return this.request('POST', '/training/control/resume');
  }

  async getTrainingRecoveryStats(): Promise<Record<string, unknown>> {
    return this.request('GET', '/recovery/stats');
  }

  async abandonRecovery(jobId: string): Promise<Record<string, unknown>> {
    return this.request('DELETE', `/recovery/abandon/${jobId}`);
  }

  // ============ Auto-Train ============

  async startAutoTrain(config: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.request('POST', '/auto-train/start', config);
  }

  async stopAutoTrain(): Promise<Record<string, unknown>> {
    return this.request('POST', '/auto-train/stop');
  }

  async getAutoTrainStatus(): Promise<Record<string, unknown>> {
    return this.request('GET', '/auto-train/status');
  }

  async listAutoTrainCheckpoints(): Promise<Record<string, unknown>[]> {
    return this.request<Record<string, unknown>[]>('GET', '/auto-train/checkpoints');
  }

  async deleteAutoTrainCheckpoint(name: string): Promise<Record<string, unknown>> {
    return this.request('DELETE', `/auto-train/checkpoints/${name}`);
  }

  async loadAutoTrainCheckpoint(name: string): Promise<Record<string, unknown>> {
    return this.request('POST', `/auto-train/checkpoints/${name}/load`);
  }

  // ============ Feedback ============

  async recordFeedback(feedback: FeedbackRecord): Promise<Record<string, unknown>> {
    return this.request('POST', '/feedback/workflow-record', feedback);
  }

  async getFeedbackStats(): Promise<Record<string, unknown>> {
    return this.request('GET', '/feedback/stats/summary');
  }

  // ============ Workflow ============

  async getWorkflowStatus(): Promise<WorkflowStatus> {
    return this.request<WorkflowStatus>('GET', '/workflow/status');
  }

  // ============ Metrics ============

  async metrics(): Promise<MetricsData> {
    return this.request<MetricsData>('GET', '/metrics');
  }

  // ============ Experiments ============

  async createExperiment(
    name: string,
    description?: string
  ): Promise<Experiment> {
    return this.request<Experiment>('POST', '/experiments', { name, description });
  }

  async listExperiments(): Promise<Experiment[]> {
    return this.request<Experiment[]>('GET', '/experiments');
  }

  async getExperiment(experimentId: string): Promise<Experiment> {
    return this.request<Experiment>('GET', `/experiments/${experimentId}`);
  }

  async logMetric(
    experimentId: string,
    metric: string,
    value: number,
    step?: number
  ): Promise<void> {
    await this.request('POST', `/experiments/${experimentId}/log_metric`, {
      metric,
      value,
      step,
    });
  }

  // ============ Datasets ============

  async listDatasets(): Promise<Record<string, unknown>[]> {
    return this.request('GET', '/datasets');
  }

  async getDataset(datasetId: string): Promise<Record<string, unknown>> {
    return this.request('GET', `/datasets/${datasetId}`);
  }

  async getDatasetStats(datasetId: string): Promise<Record<string, unknown>> {
    return this.request('GET', `/datasets/${datasetId}/stats`);
  }

  async importDatasetLocal(path: string, name?: string): Promise<Record<string, unknown>> {
    return this.request('POST', '/datasets/import/local', { path, name });
  }

  async importDatasetGitHub(repo: string, name?: string): Promise<Record<string, unknown>> {
    return this.request('POST', '/datasets/import/github', { repo, name });
  }

  async importDatasetUrl(url: string, name?: string): Promise<Record<string, unknown>> {
    return this.request('POST', '/datasets/import/url', { url, name });
  }

  // ============ Rate Limit ============

  async rateLimitStatus(): Promise<Record<string, unknown>> {
    return this.request('GET', '/rate-limit/status');
  }

  // ============ Security ============

  async getAuditLog(): Promise<Record<string, unknown>[]> {
    return this.request<Record<string, unknown>[]>('GET', '/security/audit');
  }

  async getSecurityKeys(): Promise<Record<string, unknown>[]> {
    return this.request<Record<string, unknown>[]>('GET', '/security/keys');
  }

  // ============ Auth ============

  async getToken(username: string, password: string): Promise<unknown> {
    return this.request('POST', '/auth/token', { username, password });
  }

  async refreshToken(refreshToken: string): Promise<unknown> {
    return this.request('POST', '/auth/refresh', { refresh_token: refreshToken });
  }

  // ============ Benchmark ============

  async runBenchmark(config: {
    model?: string;
    num_samples?: number;
    dataset?: string;
  }): Promise<Record<string, unknown>> {
    return this.request('POST', '/benchmark/run', config);
  }

  async runPerplexityBenchmark(config: {
    model?: string;
    dataset?: string;
  }): Promise<Record<string, unknown>> {
    return this.request('POST', '/benchmark/perplexity', config);
  }

  async getBenchmarkMetrics(): Promise<Record<string, unknown>[]> {
    return this.request<Record<string, unknown>[]>('GET', '/benchmark/metrics');
  }

  async getBenchmarkStats(): Promise<Record<string, unknown>> {
    return this.request('GET', '/benchmark/stats');
  }

  // ============ Convenience Methods ============

  async quickGenerate(prompt: string): Promise<string> {
    const result = await this.generate({ prompt });
    return result.text;
  }

  async quickChat(message: string): Promise<string> {
    const result = await this.chat({
      messages: [{ role: 'user', content: message }],
    });
    return result.message.content;
  }
}

export default SloughGPTClient;
