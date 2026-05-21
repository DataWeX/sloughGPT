export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

// ============ Federated Learning Interfaces ============

export interface LayerDelta {
  layer_name: string;
  old_weights: number[];
  new_weights: number[];
  learning_rate?: number;
  training_samples?: number;
  loss?: number;
}

export interface FederatedUpdate {
  client_id: string;
  token: string;
  model_version: number;
  layer_deltas: LayerDelta[];
  total_training_samples?: number;
  metadata?: Record<string, unknown>;
}

export interface FederatedRegistration {
  client_id: string;
  device_info?: Record<string, unknown>;
  current_model_version?: number;
}

export interface FederatedRegistrationResponse {
  client_id: string;
  token: string;
  registered: boolean;
}

export interface FederatedUpdateResponse {
  received: boolean;
  update_id: string;
  pending_updates: number;
}

export interface FederatedModelUpdate {
  version: number;
  weights: Record<string, { shape: number[]; data: number[] }>;
  is_update_available: boolean;
}

export interface FederatedStatus {
  global_version: number;
  pending_updates: number;
  registered_clients: number;
  last_aggregation: string | null;
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
  /** Ignored for wire format; server uses `POST /chat` vs `POST /chat/stream`. */
  stream?: boolean;
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
  /** Present when the API includes token counts on `POST /chat`. */
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

/**
 * Tracked HTTP training job from `GET /training/jobs` / `GET /training/jobs/{id}`.
 * When `checkpoint` is present (often after completion), it usually references native
 * `step_*.pt` with char vocab (`stoi` / `itos` / `chars`); see `docs/policies/CONTRIBUTING.md`
 * (*Checkpoint vocabulary*).
 */
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

/**
 * Body for POST /training/start (matches server `TrainingRequest`).
 *
 * Trainer `step_*.pt` on the server includes `stoi` / `itos` / `chars`; see
 * `docs/policies/CONTRIBUTING.md` (*Checkpoint vocabulary*).
 */
export interface TrainingStartPayload {
  name: string
  model: string
  /** Exactly one of ``dataset``, ``manifest_uri``, or ``dataset_ref`` is required. */
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

/** Legacy aliases ``model_name`` / ``dataset_id`` (mapped to ``model`` / ``dataset``). */
export type LegacyTrainingStartInput = Omit<
  Partial<TrainingStartPayload>,
  'model' | 'dataset' | 'name'
> & {
  model_name: string
  dataset_id: string
  name?: string
}

export interface Experiment {
  experiment_id: string;
  name: string;
  description?: string;
  metrics?: Record<string, number>;
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

  private _bodyForTrainingStart(
    input: TrainingStartPayload | LegacyTrainingStartInput
  ): Record<string, unknown> {
    const i = input as LegacyTrainingStartInput & Partial<TrainingStartPayload>
    const {
      model_name,
      dataset_id,
      model,
      dataset,
      name,
      manifest_uri,
      dataset_ref,
      epochs,
      batch_size,
      learning_rate,
      n_embed,
      n_layer,
      n_head,
      block_size,
      max_steps,
      log_interval,
      eval_interval,
    } = i
    const m = model ?? model_name
    const body: Record<string, unknown> = {}
    if (epochs !== undefined) body.epochs = epochs
    if (batch_size !== undefined) body.batch_size = batch_size
    if (learning_rate !== undefined) body.learning_rate = learning_rate
    if (n_embed !== undefined) body.n_embed = n_embed
    if (n_layer !== undefined) body.n_layer = n_layer
    if (n_head !== undefined) body.n_head = n_head
    if (block_size !== undefined) body.block_size = block_size
    if (max_steps !== undefined) body.max_steps = max_steps
    if (log_interval !== undefined) body.log_interval = log_interval
    if (eval_interval !== undefined) body.eval_interval = eval_interval
    if (m !== undefined) body.model = m
    if (name !== undefined) body.name = name
    else if (typeof m === 'string') body.name = `${m}-training`
    if (dataset_ref !== undefined) body.dataset_ref = dataset_ref
    else if (manifest_uri !== undefined) body.manifest_uri = manifest_uri
    else {
      const d = dataset ?? dataset_id
      if (d !== undefined) body.dataset = d
    }
    return body
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

  async health(): Promise<HealthStatus> {
    return this.request<HealthStatus>('GET', '/health');
  }

  async liveness(): Promise<{ status: string }> {
    return this.request('GET', '/health/live');
  }

  async readiness(): Promise<{ status: string; model_loaded: boolean }> {
    return this.request('GET', '/health/ready');
  }

  async info(): Promise<SystemInfo> {
    return this.request<SystemInfo>('GET', '/info');
  }

  async generate(request: GenerateRequest): Promise<GenerationResult> {
    this.log('info', `Generating: "${request.prompt.slice(0, 50)}..."`);
    return this.request<GenerationResult>('POST', '/generate', {
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
    const url = `${this.baseUrl}/generate/stream`;
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
      choices?: Array<{ message?: { role?: string; content?: string } }>;
    }>('POST', '/chat', {
      messages: request.messages,
      model: request.model,
      temperature: request.temperature ?? 0.8,
      max_new_tokens: request.max_new_tokens ?? 100,
      top_p: request.top_p ?? 0.9,
      top_k: request.top_k ?? 50,
    });

    let content = '';
    const choiceContent = raw.choices?.[0]?.message?.content;
    if (choiceContent != null && choiceContent !== '') {
      content = String(choiceContent);
    } else if (typeof raw.text === 'string') {
      content = raw.text;
    }

    if (typeof raw.error === 'string' && raw.error.trim() !== '' && !content.trim()) {
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
          /* ignore malformed frames */
        }
      }
    }
  }

  async batchGenerate(prompts: string[]): Promise<GenerationResult[]> {
    return this.request<GenerationResult[]>('POST', '/generate/batch', { prompts });
  }

  async listModels(): Promise<ModelInfo[]> {
    return this.request<ModelInfo[]>('GET', '/models');
  }

  async getModel(modelId: string): Promise<ModelInfo> {
    return this.request<ModelInfo>('GET', `/models/${modelId}`);
  }

  async loadModel(modelId: string): Promise<{ status: string }> {
    return this.request('POST', '/models/load', { model_id: modelId });
  }

  async metrics(): Promise<MetricsData> {
    return this.request<MetricsData>('GET', '/metrics');
  }

  /**
   * Start a tracked training job (`POST /training/start`).
   * Trainer `step_*.pt` on the server embeds char vocab for eval; see
   * `docs/policies/CONTRIBUTING.md` (*Checkpoint vocabulary*).
   */
  async startTraining(input: TrainingStartPayload | LegacyTrainingStartInput): Promise<TrainingJob> {
    return this.request<TrainingJob>('POST', '/training/start', this._bodyForTrainingStart(input));
  }

  /**
   * Poll one job (`GET /training/jobs/{id}`). Completed jobs may set `checkpoint`;
   * native `step_*.pt` embeds char vocab — `docs/policies/CONTRIBUTING.md` (*Checkpoint vocabulary*).
   */
  async getTrainingStatus(jobId: string): Promise<TrainingJob> {
    return this.request<TrainingJob>('GET', `/training/jobs/${jobId}`);
  }

  /**
   * List tracked jobs (`GET /training/jobs`). Same `checkpoint` / `step_*.pt` semantics as
   * {@link getTrainingStatus}.
   */
  async listTrainingJobs(): Promise<TrainingJob[]> {
    return this.request<TrainingJob[]>('GET', '/training/jobs');
  }

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

  async getPersonalities(): Promise<Array<{ name: string; description: string }>> {
    return this.request('GET', '/personalities');
  }

  async setPersonality(personality: string): Promise<{ status: string }> {
    return this.request('POST', '/personalities', { personality });
  }

  async listDatasets(): Promise<unknown[]> {
    return this.request('GET', '/datasets');
  }

  async getDataset(datasetId: string): Promise<unknown> {
    return this.request('GET', `/datasets/${datasetId}`);
  }

  async getDatasetStats(datasetId: string): Promise<unknown> {
    return this.request('GET', `/datasets/${datasetId}/stats`);
  }

  async inferenceGenerate(request: GenerateRequest): Promise<GenerationResult> {
    return this.request<GenerationResult>('POST', '/inference/generate', request);
  }

  async *inferenceStream(
    request: GenerateRequest
  ): AsyncGenerator<string, void, unknown> {
    const url = `${this.baseUrl}/inference/generate/stream`;
    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(request),
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
          if (data && data !== '[DONE]') yield data;
        }
      }
    }
  }

  async inferenceStats(): Promise<unknown> {
    return this.request('GET', '/inference/stats');
  }

  async inferenceBatch(prompts: string[]): Promise<unknown> {
    return this.request('POST', '/inference/batch', { prompts });
  }

  async registerModel(model: {
    name: string;
    model_type?: string;
    description?: string;
    config?: Record<string, unknown>;
  }): Promise<unknown> {
    return this.request('POST', '/registry/models', model);
  }

  async listRegisteredModels(): Promise<unknown[]> {
    return this.request('GET', '/registry/models');
  }

  async getRegisteredModel(modelId: string): Promise<unknown> {
    return this.request('GET', `/registry/models/${modelId}`);
  }

  async unregisterModel(modelId: string): Promise<unknown> {
    return this.request('DELETE', `/registry/models/${modelId}`);
  }

  async recordToRegistry(
    modelId: string,
    metrics: {
      latency_ms?: number;
      tokens_generated?: number;
      cache_hit?: boolean;
      error?: string;
    }
  ): Promise<unknown> {
    return this.request('POST', `/registry/models/${modelId}/record`, metrics);
  }

  async getRegistryMetrics(modelId: string): Promise<unknown> {
    return this.request('GET', `/registry/models/${modelId}/metrics`);
  }

  async getBestRegisteredModel(criteria: {
    metric?: string;
    order?: 'asc' | 'desc';
    model_type?: string;
  }): Promise<unknown> {
    return this.request('GET', '/registry/best', criteria);
  }

  async getRegistryStats(): Promise<unknown> {
    return this.request('GET', '/registry/stats');
  }

  async runBenchmark(config: {
    model?: string;
    num_samples?: number;
    dataset?: string;
  }): Promise<unknown> {
    return this.request('POST', '/benchmark/run', config);
  }

  async runPerplexityBenchmark(config: {
    model?: string;
    dataset?: string;
  }): Promise<unknown> {
    return this.request('POST', '/benchmark/perplexity', config);
  }

  async compareBenchmarks(modelIds: string[]): Promise<unknown> {
    return this.request('GET', `/benchmark/compare?models=${modelIds.join(',')}`);
  }

  async clearCache(): Promise<unknown> {
    return this.request('DELETE', '/cache');
  }

  async cacheStats(): Promise<unknown> {
    return this.request('GET', '/cache/stats');
  }

  async rateLimitStatus(): Promise<unknown> {
    return this.request('GET', '/rate-limit/status');
  }

  async rateLimitCheck(resource: string, cost?: number): Promise<unknown> {
    return this.request('GET', '/rate-limit/check', { resource, cost });
  }

  async getAuditLog(): Promise<unknown> {
    return this.request('GET', '/security/audit');
  }

  async getSecurityKeys(): Promise<unknown> {
    return this.request('GET', '/security/keys');
  }

  async getToken(username: string, password: string): Promise<unknown> {
    return this.request('POST', '/auth/token', { username, password });
  }

  async refreshToken(refreshToken: string): Promise<unknown> {
    return this.request('POST', '/auth/refresh', { refresh_token: refreshToken });
  }

  // ============ Federated Learning Methods ============

  async federatedRegister(registration: FederatedRegistration): Promise<FederatedRegistrationResponse> {
    this.log('info', `Registering client: ${registration.client_id}`);
    return this.request<FederatedRegistrationResponse>('POST', '/federated/register', registration);
  }

  async federatedUpdate(update: FederatedUpdate): Promise<FederatedUpdateResponse> {
    this.log('info', `Sending federated update from: ${update.client_id}`);
    return this.request<FederatedUpdateResponse>('POST', '/federated/update', {
      client_id: update.client_id,
      token: update.token,
      model_version: update.model_version,
      layer_deltas: update.layer_deltas,
      total_training_samples: update.total_training_samples,
      metadata: update.metadata,
    });
  }

  async federatedGetModel(clientId: string, currentVersion: number): Promise<FederatedModelUpdate> {
    this.log('info', `Checking for model update: ${clientId} (v${currentVersion})`);
    return this.request<FederatedModelUpdate>('GET', `/federated/model?client_id=${clientId}&current_version=${currentVersion}`);
  }

  async federatedStatus(): Promise<FederatedStatus> {
    return this.request<FederatedStatus>('GET', '/federated/status');
  }

  async federatedAggregate(): Promise<{ message: string; new_version: number; layers_updated: string[] }> {
    this.log('info', 'Triggering federated aggregation');
    return this.request('POST', '/federated/aggregate');
  }

  async federatedReset(): Promise<{ message: string }> {
    return this.request('DELETE', '/federated/reset');
  }

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
