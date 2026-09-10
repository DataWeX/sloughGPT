import { describe, it, expect, vi, beforeEach, assert } from 'vitest';
import SloughGPTClient, { SloughGPTError } from '../src/client';

const mockFetch = vi.fn();
global.fetch = mockFetch;

function createMockResponse(data: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
    headers: new Headers({ 'content-type': 'application/json' }),
  } as unknown as Response;
}

beforeEach(() => {
  mockFetch.mockReset();
});

describe('SloughGPTClient', () => {
  describe('initialization', () => {
    it('uses default base URL', () => {
      const client = new SloughGPTClient();
      expect((client as unknown as { baseUrl: string }).baseUrl).toBe('http://localhost:8000');
    });

    it('accepts custom base URL', () => {
      const client = new SloughGPTClient({ baseUrl: 'https://api.example.com' });
      expect((client as unknown as { baseUrl: string }).baseUrl).toBe('https://api.example.com');
    });

    it('strips trailing slash from base URL', () => {
      const client = new SloughGPTClient({ baseUrl: 'http://localhost:8000/' });
      expect((client as unknown as { baseUrl: string }).baseUrl).toBe('http://localhost:8000');
    });

    it('accepts API key', () => {
      const client = new SloughGPTClient({ apiKey: 'test-key' });
      expect((client as unknown as { headers: Record<string, string> }).headers['X-API-Key']).toBe('test-key');
    });

    it('uses custom timeout', () => {
      const client = new SloughGPTClient({ timeout: 60000 });
      expect((client as unknown as { timeout: number }).timeout).toBe(60000);
    });
  });

  describe('health()', () => {
    it('returns health status', async () => {
      const mockHealth = { status: 'healthy', model_loaded: true, model_type: 'gpt2' };
      mockFetch.mockResolvedValue(createMockResponse(mockHealth));

      const client = new SloughGPTClient();
      const result = await client.health();

      expect(result).toEqual(mockHealth);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/health',
        expect.objectContaining({ method: 'GET' })
      );
    });

    it('unwraps the success envelope', async () => {
      mockFetch.mockResolvedValue(
        createMockResponse({ status: 'success', data: { status: 'healthy', model_loaded: true, model_type: 'gpt2' } })
      );

      const client = new SloughGPTClient();
      const result = await client.health();

      expect(result.status).toBe('healthy');
      expect(result.model_loaded).toBe(true);
    });
  });

  describe('generate()', () => {
    it('correctly uses POST /inference/generate', async () => {
      const mockResult = { text: 'Hello world', model: 'gpt2', inference_time_ms: 150 };
      mockFetch.mockResolvedValue(createMockResponse(mockResult));

      const client = new SloughGPTClient();
      const result = await client.generate({ prompt: 'Say hello' });

      expect(result).toEqual(mockResult);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/inference/generate',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            prompt: 'Say hello',
            max_new_tokens: 100,
            temperature: 0.8,
            top_k: 50,
            top_p: 0.9,
            personality: undefined,
            model: undefined,
          }),
        })
      );
    });
  });

  describe('chat()', () => {
    it('sends POST /chat and maps message to ChatResult', async () => {
      mockFetch.mockResolvedValue(
        createMockResponse({ message: 'Hello!', model: 'gpt2-engine', tokens_generated: 3 })
      );

      const client = new SloughGPTClient();
      const result = await client.chat({
        messages: [{ role: 'user', content: 'Hi' }],
        temperature: 0.7,
        max_new_tokens: 150,
      });

      expect(result.message.content).toBe('Hello!');
      expect(result.model).toBe('gpt2-engine');
      expect(result.tokens_generated).toBe(3);
      expect(mockFetch.mock.calls[0][0]).toContain('/chat');
    });
  });

  describe('souls', () => {
    it('lists souls', async () => {
      const mockSouls = [{ name: 'friendly', description: 'A friendly soul', traits: { warmth: 0.8 } }];
      mockFetch.mockResolvedValue(createMockResponse(mockSouls));

      const client = new SloughGPTClient();
      const result = await client.listSouls();

      expect(result).toEqual(mockSouls);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/souls',
        expect.objectContaining({ method: 'GET' })
      );
    });

    it('switches soul', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'switched' }));

      const client = new SloughGPTClient();
      await client.switchSoul('friendly');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/souls/switch',
        expect.objectContaining({ method: 'POST', body: JSON.stringify({ name: 'friendly' }) })
      );
    });
  });

  describe('knowledge', () => {
    it('lists knowledge', async () => {
      const mockItems = [{ id: 'k1', content: 'fact' }];
      mockFetch.mockResolvedValue(createMockResponse(mockItems));

      const client = new SloughGPTClient();
      const result = await client.listKnowledge();

      expect(result).toEqual(mockItems);
    });

    it('searches knowledge', async () => {
      mockFetch.mockResolvedValue(createMockResponse([{ id: 'k1', content: 'fact', relevance: 0.9 }]));

      const client = new SloughGPTClient();
      const result = await client.searchKnowledge('fact');

      expect(result).toHaveLength(1);
      expect(mockFetch.mock.calls[0][0]).toContain('/knowledge/search?q=fact');
    });
  });

  describe('tokenizer', () => {
    it('gets stats', async () => {
      const mockStats = { vocab_size: 50257, num_tokens: 100000 };
      mockFetch.mockResolvedValue(createMockResponse(mockStats));

      const client = new SloughGPTClient();
      const result = await client.getTokenizerStats();

      expect(result.vocab_size).toBe(50257);
      expect(mockFetch.mock.calls[0][0]).toContain('/tokenizer/stats');
    });
  });

  describe('system', () => {
    it('gets metrics', async () => {
      const mockMetrics = { cpu_percent: 45, memory_percent: 60, disk_percent: 70, uptime_seconds: 3600, gpu_available: false };
      mockFetch.mockResolvedValue(createMockResponse(mockMetrics));

      const client = new SloughGPTClient();
      const result = await client.getSystemMetrics();

      expect(result.cpu_percent).toBe(45);
      expect(mockFetch.mock.calls[0][0]).toContain('/system/metrics');
    });
  });

  describe('workflow', () => {
    it('gets workflow status', async () => {
      const mockStatus = { status: 'active', active: true, feedback_count: 42 };
      mockFetch.mockResolvedValue(createMockResponse(mockStatus));

      const client = new SloughGPTClient();
      const result = await client.getWorkflowStatus();

      expect(result.feedback_count).toBe(42);
      expect(mockFetch.mock.calls[0][0]).toContain('/workflow/status');
    });
  });

  describe('quick methods', () => {
    it('quickGenerate returns just the text', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ text: 'Simplified response', model: 'gpt2' }));

      const client = new SloughGPTClient();
      const result = await client.quickGenerate('Hello');

      expect(result).toBe('Simplified response');
    });

    it('quickChat returns just the message content', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ text: 'Quick reply', model: 'gpt2-engine' }));

      const client = new SloughGPTClient();
      const result = await client.quickChat('Hello');

      expect(result).toBe('Quick reply');
    });
  });

  describe('error handling', () => {
    it('throws SloughGPTError on non-ok response', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ detail: 'Not found' }, 404));

      const client = new SloughGPTClient();
      await expect(client.health()).rejects.toThrow(SloughGPTError);
    });

    it('includes status code in error', async () => {
      mockFetch.mockResolvedValue(createMockResponse({}, 500));

      const client = new SloughGPTClient();
      try {
        await client.health();
        expect.fail('Should have thrown');
      } catch (e) {
        expect((e as SloughGPTError).statusCode).toBe(500);
      }
    });

    it('throws on timeout', async () => {
      mockFetch.mockImplementation(() => new Promise((_, reject) => {
        const error = new Error('Aborted');
        (error as { name: string }).name = 'AbortError';
        reject(error);
      }));

      const client = new SloughGPTClient({ timeout: 1 });
      await expect(client.health()).rejects.toThrow('Request timeout');
    });
  });

  describe('metrics()', () => {
    it('returns metrics data', async () => {
      const mockMetrics = { requests_today: 100, tokens_today: 5000, cache_hit_rate: 0.35 };
      mockFetch.mockResolvedValue(createMockResponse(mockMetrics));

      const client = new SloughGPTClient();
      const result = await client.metrics();

      expect(result).toEqual(mockMetrics);
    });
  });

  describe('experiments', () => {
    it('creates experiment', async () => {
      const mockExp = { experiment_id: 'exp-1', name: 'Test', description: 'A test' };
      mockFetch.mockResolvedValue(createMockResponse(mockExp));

      const client = new SloughGPTClient();
      const result = await client.createExperiment('Test', 'A test');

      expect(result).toEqual(mockExp);
    });
  });

  describe('training', () => {
    it('starts training with canonical body', async () => {
      const job = { id: 'job_1', name: 'run-a', model: 'sloughgpt', dataset: 'openwebtext', status: 'running' as const, progress: 0 };
      mockFetch.mockResolvedValue(createMockResponse(job));

      const client = new SloughGPTClient();
      const result = await client.startTraining({ name: 'run-a', model: 'sloughgpt', dataset: 'openwebtext', epochs: 3 });

      expect(result.id).toBe('job_1');
      expect(mockFetch.mock.calls[0][0]).toContain('/training/start');
    });
  });

  describe('session', () => {
    it('saves session context', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'stored' }));

      const client = new SloughGPTClient();
      const result = await client.saveSessionContext('sess-1', { context: 'test' });

      expect(mockFetch.mock.calls[0][0]).toContain('/session/sess-1/context');
    });

    it('gets session messages', async () => {
      const mockMessages = [{ role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello!' }];
      mockFetch.mockResolvedValue(createMockResponse(mockMessages));

      const client = new SloughGPTClient();
      const result = await client.getSessionMessages('sess-1');

      expect(result).toEqual(mockMessages);
      expect(mockFetch.mock.calls[0][0]).toContain('/session/sess-1/messages');
    });
  });

  describe('souls (extended)', () => {
    it('gets current soul', async () => {
      const mockSoul = { name: 'friendly', description: 'A friendly soul' };
      mockFetch.mockResolvedValue(createMockResponse(mockSoul));

      const client = new SloughGPTClient();
      const result = await client.getCurrentSoul();

      expect(result.name).toBe('friendly');
      expect(mockFetch.mock.calls[0][0]).toContain('/souls/current');
    });

    it('switches soul with checkpoint', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'switched' }));

      const client = new SloughGPTClient();
      await client.switchSoul('friendly', 'ckpt-v2');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/souls/switch',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ name: 'friendly', checkpoint_name: 'ckpt-v2' }),
        })
      );
    });
  });

  describe('knowledge (extended)', () => {
    it('adds knowledge', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ id: 'k_new', status: 'added' }));

      const client = new SloughGPTClient();
      const result = await client.addKnowledge('Paris is capital', 'geo');

      expect(mockFetch.mock.calls[0][0]).toContain('/knowledge');
      expect(mockFetch.mock.calls[0][1].body).toContain('Paris is capital');
    });

    it('deletes knowledge', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'deleted' }));

      const client = new SloughGPTClient();
      await client.deleteKnowledge('k1');

      expect(mockFetch.mock.calls[0][0]).toContain('/knowledge/k1');
      expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
    });

    it('gets knowledge stats', async () => {
      const mockStats = { total_facts: 42, total_topics: 5 };
      mockFetch.mockResolvedValue(createMockResponse(mockStats));

      const client = new SloughGPTClient();
      const result = await client.getKnowledgeStats();

      expect(result.total_facts).toBe(42);
      expect(mockFetch.mock.calls[0][0]).toContain('/knowledge/stats');
    });

    it('gets knowledge topics', async () => {
      mockFetch.mockResolvedValue(createMockResponse(['geo', 'science']));

      const client = new SloughGPTClient();
      const result = await client.getKnowledgeTopics();

      expect(result).toEqual(['geo', 'science']);
      expect(mockFetch.mock.calls[0][0]).toContain('/knowledge/topics');
    });

    it('ingests knowledge URL', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ files_imported: 1 }));

      const client = new SloughGPTClient();
      await client.ingestKnowledgeUrl('https://example.com');

      expect(mockFetch.mock.calls[0][0]).toContain('/knowledge/ingest-url');
    });
  });

  describe('tokenizer (extended)', () => {
    it('tokenizes text', async () => {
      const mockTokens = { tokens: [123, 456], token_count: 2 };
      mockFetch.mockResolvedValue(createMockResponse(mockTokens));

      const client = new SloughGPTClient();
      const result = await client.tokenize('hello world');

      expect(result.token_count).toBe(2);
      expect(mockFetch.mock.calls[0][0]).toContain('/tokenizer/tokenize');
    });

    it('trains tokenizer', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'trained', vocab_size: 32000 }));

      const client = new SloughGPTClient();
      const result = await client.trainTokenizer('training text', 32000);

      expect(result.vocab_size).toBe(32000);
      expect(mockFetch.mock.calls[0][0]).toContain('/tokenizer/train');
    });
  });

  describe('system (extended)', () => {
    it('gets system info', async () => {
      const mockInfo = { python_version: '3.10.0', platform: 'linux', cpu_count: 8 };
      mockFetch.mockResolvedValue(createMockResponse(mockInfo));

      const client = new SloughGPTClient();
      const result = await client.getSystemInfo();

      expect(result.platform).toBe('linux');
      expect(mockFetch.mock.calls[0][0]).toContain('/system/info');
    });

    it('gets system disk', async () => {
      const mockDisk = { total_gb: 500, used_gb: 250, free_gb: 250 };
      mockFetch.mockResolvedValue(createMockResponse(mockDisk));

      const client = new SloughGPTClient();
      const result = await client.getSystemDisk();

      expect(result.total_gb).toBe(500);
      expect(mockFetch.mock.calls[0][0]).toContain('/system/disk');
    });
  });

  describe('companion', () => {
    it('gets companion prompt', async () => {
      const mockPrompt = { prompt: 'You are a friendly assistant' };
      mockFetch.mockResolvedValue(createMockResponse(mockPrompt));

      const client = new SloughGPTClient();
      const result = await client.getCompanionPrompt();

      expect(result.prompt).toContain('friendly');
      expect(mockFetch.mock.calls[0][0]).toContain('/companion/prompt');
    });

    it('lists companion presets', async () => {
      const mockPresets = [{ name: 'friendly', description: 'Warm tone' }];
      mockFetch.mockResolvedValue(createMockResponse(mockPresets));

      const client = new SloughGPTClient();
      const result = await client.listCompanionPresets();

      expect(result).toEqual(mockPresets);
      expect(mockFetch.mock.calls[0][0]).toContain('/companion/presets');
    });

    it('sets personality via companion endpoint', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'set' }));

      const client = new SloughGPTClient();
      await client.setPersonality('friendly');

      expect(mockFetch.mock.calls[0][0]).toContain('/companion/personality');
    });
  });

  describe('training control', () => {
    it('lists training jobs', async () => {
      const mockJobs = [{ id: 'job_1', status: 'running' }];
      mockFetch.mockResolvedValue(createMockResponse(mockJobs));

      const client = new SloughGPTClient();
      const result = await client.listTrainingJobs();

      expect(result).toEqual(mockJobs);
      expect(mockFetch.mock.calls[0][0]).toContain('/training/jobs');
    });

    it('gets training status for a job', async () => {
      const mockStatus = { id: 'job_1', status: 'running', progress: 50 };
      mockFetch.mockResolvedValue(createMockResponse(mockStatus));

      const client = new SloughGPTClient();
      const result = await client.getTrainingStatus('job_1');

      expect(result.status).toBe('running');
      expect(mockFetch.mock.calls[0][0]).toContain('/training/jobs/job_1');
    });

    it('stops training', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'stopped' }));
      const client = new SloughGPTClient();
      await client.stopTraining();
      expect(mockFetch.mock.calls[0][0]).toContain('/training/control/stop');
    });

    it('pauses training', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'paused' }));
      const client = new SloughGPTClient();
      await client.pauseTraining();
      expect(mockFetch.mock.calls[0][0]).toContain('/training/control/pause');
    });

    it('resumes training', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'resumed' }));
      const client = new SloughGPTClient();
      await client.resumeTraining();
      expect(mockFetch.mock.calls[0][0]).toContain('/training/control/resume');
    });

    it('deletes training job', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'deleted' }));
      const client = new SloughGPTClient();
      await client.deleteTrainingJob('job-1');
      expect(mockFetch.mock.calls[0][0]).toContain('/training/jobs/job-1');
      expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
    });

    it('gets training recovery stats', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ checkpoint_exists: true }));
      const client = new SloughGPTClient();
      const result = await client.getTrainingRecoveryStats();
      expect(result.checkpoint_exists).toBe(true);
      expect(mockFetch.mock.calls[0][0]).toContain('/recovery/stats');
    });
  });

  describe('auto-train', () => {
    it('starts auto-train', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'started' }));
      const client = new SloughGPTClient();
      await client.startAutoTrain({ soul: 'friendly' });
      expect(mockFetch.mock.calls[0][0]).toContain('/training/start');
    });

    it('stops auto-train', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'stopped' }));
      const client = new SloughGPTClient();
      await client.stopAutoTrain();
      expect(mockFetch.mock.calls[0][0]).toContain('/training/stop');
    });

    it('gets auto-train status', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ phase: 'TRAIN' }));
      const client = new SloughGPTClient();
      const result = await client.getAutoTrainStatus();
      expect(result.phase).toBe('TRAIN');
      expect(mockFetch.mock.calls[0][0]).toContain('/training/status');
    });

    it('lists checkpoints', async () => {
      mockFetch.mockResolvedValue(createMockResponse([{ name: 'ckpt-1' }]));
      const client = new SloughGPTClient();
      const result = await client.listAutoTrainCheckpoints();
      expect(result).toHaveLength(1);
      expect(mockFetch.mock.calls[0][0]).toContain('/training/checkpoints');
    });

    it('deletes checkpoint', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'deleted' }));
      const client = new SloughGPTClient();
      await client.deleteAutoTrainCheckpoint('ckpt-1');
      expect(mockFetch.mock.calls[0][0]).toContain('/training/checkpoints/ckpt-1');
      expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
    });

    it('loads checkpoint', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'loaded' }));
      const client = new SloughGPTClient();
      await client.loadAutoTrainCheckpoint('ckpt-1');
      expect(mockFetch.mock.calls[0][0]).toContain('/training/checkpoints/ckpt-1/load');
    });
  });

  describe('feedback', () => {
    it('records feedback', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'recorded' }));
      const client = new SloughGPTClient();
      await client.recordFeedback('s1', 'm1', 1);
      expect(mockFetch.mock.calls[0][0]).toContain('/feedback/workflow-record');
    });

    it('gets feedback stats', async () => {
      const mockStats = { total: 100, positive: 60, negative: 40 };
      mockFetch.mockResolvedValue(createMockResponse(mockStats));
      const client = new SloughGPTClient();
      const result = await client.getFeedbackStats();
      expect(result.total).toBe(100);
      expect(mockFetch.mock.calls[0][0]).toContain('/feedback/stats/summary');
    });
  });

  describe('models (extended)', () => {
    it('unloads model', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'unloaded' }));
      const client = new SloughGPTClient();
      await client.unloadModel();
      expect(mockFetch.mock.calls[0][0]).toContain('/models/unload');
    });

    it('gets current model', async () => {
      const mockModel = { name: 'gpt2', loaded: true, type: 'huggingface' };
      mockFetch.mockResolvedValue(createMockResponse(mockModel));
      const client = new SloughGPTClient();
      const result = await client.getCurrentModel();
      expect(result.name).toBe('gpt2');
      expect(mockFetch.mock.calls[0][0]).toContain('/models/current');
    });

    it('listModels unwraps the success envelope', async () => {
      mockFetch.mockResolvedValue(
        createMockResponse({ status: 'success', data: [{ model_id: 'gpt2', name: 'GPT-2' }] })
      );
      const client = new SloughGPTClient();
      const result = await client.listModels();
      expect(result).toEqual([{ model_id: 'gpt2', name: 'GPT-2' }]);
    });
  });

  describe('auth', () => {
    it('exchanges an API key for a JWT token', async () => {
      mockFetch.mockResolvedValue(
        createMockResponse({ status: 'success', data: { access_token: 'jwt', token_type: 'bearer' } })
      );
      const client = new SloughGPTClient();
      const result = await client.getToken('secret');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/auth/token',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ api_key: 'secret' }),
        })
      );
      expect(result).toEqual({ status: 'success', data: { access_token: 'jwt', token_type: 'bearer' } });
    });
  });

  describe('datasets (extended)', () => {
    it('imports dataset from local path', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ files_imported: 5 }));
      const client = new SloughGPTClient();
      const result = await client.importDatasetLocal('/path/to/data', 'my-dataset');
      expect(result.files_imported).toBe(5);
      expect(mockFetch.mock.calls[0][0]).toContain('/datasets/import/local');
    });

    it('imports dataset from GitHub', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ files_imported: 10 }));
      const client = new SloughGPTClient();
      const result = await client.importDatasetGitHub('user/repo', 'gh-dataset');
      expect(result.files_imported).toBe(10);
      expect(mockFetch.mock.calls[0][0]).toContain('/datasets/import/github');
    });

    it('imports dataset from URL', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ files_imported: 1 }));
      const client = new SloughGPTClient();
      const result = await client.importDatasetUrl('https://example.com', 'url-dataset');
      expect(result.files_imported).toBe(1);
      expect(mockFetch.mock.calls[0][0]).toContain('/datasets/import/url');
    });
  });

  describe('benchmark', () => {
    it('gets benchmark metrics', async () => {
      const mockMetrics = { avg_latency_ms: 150, tokens_per_second: 20 };
      mockFetch.mockResolvedValue(createMockResponse(mockMetrics));
      const client = new SloughGPTClient();
      const result = await client.getBenchmarkMetrics();
      expect(result.avg_latency_ms).toBe(150);
      expect(mockFetch.mock.calls[0][0]).toContain('/benchmark/metrics');
    });

    it('gets benchmark stats', async () => {
      const mockStats = { total_runs: 10, success_rate: 0.95 };
      mockFetch.mockResolvedValue(createMockResponse(mockStats));
      const client = new SloughGPTClient();
      const result = await client.getBenchmarkStats();
      expect(result.total_runs).toBe(10);
      expect(mockFetch.mock.calls[0][0]).toContain('/benchmark/stats');
    });
  });

  describe('security', () => {
    it('gets audit log', async () => {
      const mockLog = [{ action: 'model_loaded', timestamp: '2026-01-01T00:00:00Z' }];
      mockFetch.mockResolvedValue(createMockResponse(mockLog));
      const client = new SloughGPTClient();
      const result = await client.getAuditLog();
      expect(result).toEqual(mockLog);
      expect(mockFetch.mock.calls[0][0]).toContain('/security/audit');
    });

    it('gets security keys', async () => {
      const mockKeys = [{ key_id: 'k1' }];
      mockFetch.mockResolvedValue(createMockResponse(mockKeys));
      const client = new SloughGPTClient();
      const result = await client.getSecurityKeys();
      expect(result).toEqual(mockKeys);
      expect(mockFetch.mock.calls[0][0]).toContain('/security/keys');
    });
  });

  describe('model registry', () => {
    it('lists registry models (unwraps models field)', async () => {
      mockFetch.mockResolvedValue(
        createMockResponse({ status: 'success', data: { models: [{ model_id: 'gpt2' }], count: 1 } })
      );
      const client = new SloughGPTClient();
      const result = await client.listRegistryModels();
      expect(result).toEqual([{ model_id: 'gpt2' }]);
      expect(mockFetch.mock.calls[0][0]).toContain('/registry/models');
    });

    it('gets single registry model (unwraps envelope)', async () => {
      const mockModel = { model_id: 'gpt2', params: 124_000_000 };
      mockFetch.mockResolvedValue(createMockResponse({ status: 'success', data: mockModel }));
      const client = new SloughGPTClient();
      const result = await client.getRegistryModel('gpt2');
      expect(result).toEqual(mockModel);
      expect(mockFetch.mock.calls[0][0]).toContain('/registry/models/gpt2');
    });

    it('gets best registry model (unwraps envelope)', async () => {
      const mockBest = { models: 2, loaded: 1 };
      mockFetch.mockResolvedValue(createMockResponse({ status: 'success', data: mockBest }));
      const client = new SloughGPTClient();
      const result = await client.getRegistryBest();
      expect(result).toEqual(mockBest);
      expect(mockFetch.mock.calls[0][0]).toContain('/registry/best');
    });

    it('gets registry stats (unwraps envelope)', async () => {
      const mockStats = { models: 5, best_score: 0.9 };
      mockFetch.mockResolvedValue(createMockResponse({ status: 'success', data: mockStats }));
      const client = new SloughGPTClient();
      const result = await client.getRegistryStats();
      expect(result).toEqual(mockStats);
      expect(mockFetch.mock.calls[0][0]).toContain('/registry/stats');
    });
  });

  describe('detailed health', () => {
    it('returns detailed health', async () => {
      const mockHealth = { status: 'healthy', model: { name: 'gpt2' }, system: { cpu: 45, memory: 60 } };
      mockFetch.mockResolvedValue(createMockResponse(mockHealth));
      const client = new SloughGPTClient();
      const result = await client.detailedHealth();
      expect(result.status).toBe('healthy');
      expect(mockFetch.mock.calls[0][0]).toContain('/health/detailed');
    });
  });

  describe('generate stream', () => {
    it('yields tokens from SSE envelopes', async () => {
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode('data: {"stream":"generate","status":"working","data":{"token":"Hello"}}\n\n'));
          controller.enqueue(encoder.encode('data: {"stream":"generate","status":"working","data":{"token":" world"}}\n\n'));
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        },
      });
      mockFetch.mockResolvedValue({ ok: true, body: stream, status: 200, statusText: 'OK', headers: new Headers() });

      const client = new SloughGPTClient();
      const tokens: string[] = [];
      for await (const token of client.generateStream({ prompt: 'hello' })) {
        tokens.push(token);
      }
      expect(tokens).toEqual(['Hello', ' world']);
    });
  });

  describe('regenerate', () => {
    it('calls regenerate endpoint', async () => {
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode('data: {"stream":"chat","phase":"STREAMING","data":{"token":"regenerated"},"status":"working"}\n\n'));
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        },
      });
      mockFetch.mockResolvedValue({ ok: true, body: stream, status: 200, statusText: 'OK', headers: new Headers() });

      const client = new SloughGPTClient();
      const tokens: string[] = [];
      for await (const token of client.regenerateStream('sess-1')) {
        tokens.push(token);
      }
      expect(tokens).toEqual(['regenerated']);
      expect(mockFetch.mock.calls[0][0]).toContain('/session/sess-1/regenerate');
    });
  });

  describe('exportTrainingHistory()', () => {
    it('calls GET /settings/training/history/export', async () => {
      const mockData = { format: 'json', outcomes: [], count: 0 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.exportTrainingHistory('json', 10);

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/history/export?format=json&limit=10');
    });
  });

  describe('generateModelCard()', () => {
    it('calls POST /settings/model-card', async () => {
      const mockData = { card: { model_name: 'test' }, markdown: '# test' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.generateModelCard('test', { base_model: 'gpt2' });

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/model-card');
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('getDashboardSummary()', () => {
    it('calls GET /dashboard/summary', async () => {
      const mockData = { health: { model_loaded: true }, active_processes: 0, services: { total: 2, healthy: 2 } };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getDashboardSummary();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/dashboard/summary');
    });
  });

  describe('compareTrainingRuns()', () => {
    it('calls GET /settings/training/compare', async () => {
      const mockData = { run_a: {}, run_b: {}, differences: {}, a_wins: 1, b_wins: 0 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.compareTrainingRuns('run-1', 'run-2');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/compare?run_a=run-1&run_b=run-2');
    });
  });

  describe('getBatchTrainingStatus()', () => {
    it('calls GET /settings/training/batch-status', async () => {
      const mockData = { jobs: [], summary: { total: 0, running: 0, queued: 0, completed: 0, failed: 0 } };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getBatchTrainingStatus();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/batch-status');
    });
  });

  describe('listTrainingPresets()', () => {
    it('calls GET /settings/training/presets', async () => {
      const mockData = { presets: [{ name: 'quick-finetune', model: 'gpt2' }] };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.listTrainingPresets();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/presets');
    });
  });

  describe('getTrainingPreset()', () => {
    it('calls GET /settings/training/presets/:name', async () => {
      const mockData = { name: 'Quick Fine-Tune', model: 'gpt2' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getTrainingPreset('quick-finetune');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/presets/quick-finetune');
    });
  });

  describe('applyTrainingPreset()', () => {
    it('calls POST /settings/training/presets/:name/apply', async () => {
      const mockData = { preset: 'quick-finetune', applied: {} };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.applyTrainingPreset('quick-finetune');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/presets/quick-finetune/apply');
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('getTrainingRun()', () => {
    it('calls GET /settings/training/runs/:runId', async () => {
      const mockData = { run_id: 'run-1', model: 'gpt2', quality_score: 0.85 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getTrainingRun('run-1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs/run-1');
    });
  });

  describe('deleteTrainingRun()', () => {
    it('calls DELETE /settings/training/runs/:runId', async () => {
      const mockData = { deleted: true, run_id: 'run-1' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.deleteTrainingRun('run-1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs/run-1');
      expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
    });
  });

  describe('filterTrainingRuns()', () => {
    it('calls GET /settings/training/runs with params', async () => {
      const mockData = { runs: [], count: 0 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.filterTrainingRuns({ model: 'gpt2', limit: 10 });

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs?model=gpt2&limit=10');
    });
  });

  describe('clearTrainingHistory()', () => {
    it('calls POST /settings/training/history/clear', async () => {
      const mockData = { cleared: true, removed_count: 5 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.clearTrainingHistory();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/history/clear');
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('addRunTag()', () => {
    it('calls POST /settings/training/runs/:runId/tags', async () => {
      const mockData = { run_id: 'run-1', tags: ['best'] };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.addRunTag('run-1', 'best');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs/run-1/tags?tag=best');
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('removeRunTag()', () => {
    it('calls DELETE /settings/training/runs/:runId/tags/:tag', async () => {
      const mockData = { run_id: 'run-1', tags: [] };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.removeRunTag('run-1', 'best');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs/run-1/tags/best');
      expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
    });
  });

  describe('setRunNotes()', () => {
    it('calls PUT /settings/training/runs/:runId/notes', async () => {
      const mockData = { run_id: 'run-1', notes: 'test note' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.setRunNotes('run-1', 'test note');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs/run-1/notes?notes=test');
      expect(mockFetch.mock.calls[0][1].method).toBe('PUT');
    });
  });

  describe('getAllTags()', () => {
    it('calls GET /settings/training/tags', async () => {
      const mockData = { tags: ['best', 'production'] };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getAllTags();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/tags');
    });
  });

  describe('getRunsByTag()', () => {
    it('calls GET /settings/training/tags/:tag', async () => {
      const mockData = { runs: [], count: 0, tag: 'best' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getRunsByTag('best');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/tags/best');
    });
  });

  describe('exportTrainingRun()', () => {
    it('calls GET /settings/training/runs/:runId/export', async () => {
      const mockData = { run_id: 'run-1', format: 'json', content: '{}' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.exportTrainingRun('run-1', 'json');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs/run-1/export?format=json');
    });
  });

  describe('toggleBookmark()', () => {
    it('calls POST /settings/training/runs/:runId/bookmark', async () => {
      const mockData = { run_id: 'run-1', bookmarked: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.toggleBookmark('run-1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs/run-1/bookmark');
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('getBookmarkedRuns()', () => {
    it('calls GET /settings/training/bookmarks', async () => {
      const mockData = { runs: [], count: 0 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getBookmarkedRuns();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/bookmarks');
    });
  });

  describe('duplicateTrainingRun()', () => {
    it('calls POST /settings/training/runs/:runId/duplicate', async () => {
      const mockData = { run_id: 'new-run', model: 'gpt2' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.duplicateTrainingRun('run-1', 'new-run');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs/run-1/duplicate?new_run_id=new-run');
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('bulkDeleteRuns()', () => {
    it('calls POST /settings/training/runs/bulk/delete', async () => {
      const mockData = { deleted_count: 2, requested: 2 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.bulkDeleteRuns(['run-1', 'run-2']);

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs/bulk/delete');
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('bulkAddTag()', () => {
    it('calls POST /settings/training/runs/bulk/tag', async () => {
      const mockData = { updated_count: 2, tag: 'best' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.bulkAddTag(['run-1', 'run-2'], 'best');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs/bulk/tag');
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('bulkBookmark()', () => {
    it('calls POST /settings/training/runs/bulk/bookmark', async () => {
      const mockData = { updated_count: 2, bookmarked: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.bulkBookmark(['run-1', 'run-2'], true);

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/runs/bulk/bookmark');
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('getAutoTrainSettingsStatus()', () => {
    it('calls GET /settings/training/auto-train/status', async () => {
      const mockData = { enabled: true, threshold: 10 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getAutoTrainSettingsStatus();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/auto-train/status');
    });
  });

  describe('updateAutoTrainSettingsConfig()', () => {
    it('calls PATCH /settings/training/auto-train/config', async () => {
      const mockData = { enabled: true, threshold: 20 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.updateAutoTrainSettingsConfig({ threshold: 20 });

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/training/auto-train/config');
      expect(mockFetch.mock.calls[0][1].method).toBe('PATCH');
    });
  });

  describe('Security Keys', () => {
    it('listSecurityKeys calls GET /security/keys', async () => {
      const mockData = [{ id: 'k1' }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getSecurityKeys();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/security/keys');
    });

    it('createSecurityKey calls POST /security/keys', async () => {
      const mockData = { id: 'k2', name: 'test' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.createSecurityKey('test', ['read'], 30);

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });

    it('deleteSecurityKey calls DELETE /security/keys/{keyId}', async () => {
      const mockData = { deleted: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.deleteSecurityKey('k1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
    });

    it('rotateSecurityKey calls POST /security/keys/{keyId}/rotate', async () => {
      const mockData = { id: 'k1', new_secret: 's3cret' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.rotateSecurityKey('k1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });

    it('validateSecurityKey calls POST /security/keys/validate', async () => {
      const mockData = { valid: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.validateSecurityKey('my-key');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('Tenants', () => {
    it('listTenants calls GET /tenants', async () => {
      const mockData = [{ id: 't1' }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.listTenants();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/tenants');
    });

    it('getTenant calls GET /tenants/{tenantId}', async () => {
      const mockData = { id: 't1', name: 'acme' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getTenant('t1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/tenants/t1');
    });

    it('createTenant calls POST /tenants', async () => {
      const mockData = { id: 't2', name: 'new' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.createTenant('new', { plan: 'pro' });

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });

    it('deleteTenant calls DELETE /tenants/{tenantId}', async () => {
      const mockData = { deleted: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.deleteTenant('t1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
    });

    it('getTenantStats calls GET /tenants/{tenantId}/stats', async () => {
      const mockData = { users: 10, workspaces: 3 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getTenantStats('t1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/tenants/t1/stats');
    });
  });

  describe('Profiles', () => {
    it('listProfiles calls GET /profiles', async () => {
      const mockData = [{ id: 'p1' }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.listProfiles();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/profiles');
    });

    it('getProfile calls GET /profiles/{profileId}', async () => {
      const mockData = { id: 'p1', name: 'default' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getProfile('p1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/profiles/p1');
    });

    it('applyProfile calls POST /profiles/apply', async () => {
      const mockData = { applied: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.applyProfile('p1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });

    it('getActiveProfile calls GET /profiles/active', async () => {
      const mockData = { id: 'p1' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getActiveProfile();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/profiles/active');
    });

    it('recommendProfile calls GET /profiles/recommend', async () => {
      const mockData = { profile_id: 'p2', reason: 'best match' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.recommendProfile();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/profiles/recommend');
    });
  });

  describe('Workspaces', () => {
    it('listWorkspaces calls GET /workspaces', async () => {
      const mockData = [{ id: 'w1' }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.listWorkspaces();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/workspaces');
    });

    it('getWorkspace calls GET /workspaces/{workspaceId}', async () => {
      const mockData = { id: 'w1', name: 'team' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getWorkspace('w1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/workspaces/w1');
    });

    it('createWorkspace calls POST /workspaces', async () => {
      const mockData = { id: 'w2', name: 'new' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.createWorkspace('new', { description: 'test' });

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });

    it('addWorkspaceMember calls POST /workspaces/{id}/members', async () => {
      const mockData = { added: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.addWorkspaceMember('w1', 'user-1', 'admin');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('Auth', () => {
    it('getToken calls POST /auth/token', async () => {
      const mockData = { access_token: 'jwt-123', token_type: 'bearer' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getToken('my-api-key');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });
  });

  describe('Health', () => {
    it('health calls GET /health', async () => {
      const mockData = { status: 'healthy', model_loaded: true, model_type: 'gpt2' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.health();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/health');
    });

    it('liveness calls GET /health/live', async () => {
      const mockData = { status: 'alive' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.liveness();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/health/live');
    });

    it('readiness calls GET /health/ready', async () => {
      const mockData = { status: 'ready', model_loaded: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.readiness();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/health/ready');
    });

    it('detailedHealth calls GET /health/detailed', async () => {
      const mockData = { status: 'ok', uptime: 1000 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.detailedHealth();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/health/detailed');
    });

    it('info calls GET /info', async () => {
      const mockData = { name: 'sloughgpt', version: '1.0', model: { type: 'gpt2', loaded: true } };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.info();

      expect(result.version).toBe('1.0');
      expect(result.model.type).toBe('gpt2');
      expect(mockFetch.mock.calls[0][0]).toContain('/info');
    });
  });

  describe('Settings', () => {
    it('getSettings calls GET /settings', async () => {
      const mockData = { generation: { temperature: 0.7 } };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getSettings();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings');
    });

    it('getGenerationSettings calls GET /settings/generation', async () => {
      const mockData = { temperature: 0.8, top_p: 0.9 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getGenerationSettings();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/generation');
    });

    it('updateGenerationSettings calls PATCH /settings/generation', async () => {
      const mockData = { temperature: 0.5 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.updateGenerationSettings({ temperature: 0.5 });

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('PATCH');
    });

    it('getVoiceSettings calls GET /settings/voice', async () => {
      const mockData = { noise_gate_db: -40 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getVoiceSettings();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/voice');
    });

    it('updateVoiceSettings calls PATCH /settings/voice', async () => {
      const mockData = { noise_gate_db: -35 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.updateVoiceSettings({ noise_gate_db: -35 });

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('PATCH');
    });

    it('resetSettings calls POST /settings/reset', async () => {
      const mockData = { reset: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.resetSettings();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });

    it('getAdaptiveInsights calls GET /settings/adaptive/insights', async () => {
      const mockData = { exploration_rate: 0.3, confidence: 0.8 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getAdaptiveInsights();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/settings/adaptive/insights');
    });
  });

  describe('Models', () => {
    it('listModels calls GET /models', async () => {
      const mockData = [{ model_id: 'gpt2', name: 'GPT-2' }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.listModels();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/models');
    });

    it('loadModel calls POST /models/load', async () => {
      const mockData = { status: 'loaded', model_id: 'gpt2' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.loadModel('gpt2');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });

    it('unloadModel calls POST /models/unload', async () => {
      const mockData = { status: 'unloaded' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.unloadModel();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });

    it('getCurrentModel calls GET /models/current', async () => {
      const mockData = { model_id: 'gpt2', loaded: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getCurrentModel();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/models/current');
    });
  });

  describe('Sessions', () => {
    it('createSession calls POST /chat/sessions', async () => {
      const mockData = { session_id: 's1', created_at: '2026-09-10' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.createSession();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });

    it('listSessions calls GET /chat/sessions', async () => {
      const mockData = [{ session_id: 's1' }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.listSessions();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/chat/sessions');
    });

    it('getSession calls GET /chat/sessions/{id}', async () => {
      const mockData = { session_id: 's1', messages: [] };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getSession('s1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/chat/sessions/s1');
    });

    it('deleteSession calls DELETE /chat/sessions/{id}', async () => {
      const mockData = { deleted: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.deleteSession('s1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
    });
  });

  describe('Knowledge', () => {
    it('listKnowledge calls GET /knowledge', async () => {
      const mockData = [{ id: 'k1', content: 'test' }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.listKnowledge();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/knowledge');
    });

    it('addKnowledge calls POST /knowledge', async () => {
      const mockData = { id: 'k2', content: 'new fact' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.addKnowledge('new fact', 'ai');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });

    it('deleteKnowledge calls DELETE /knowledge/{id}', async () => {
      const mockData = { deleted: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.deleteKnowledge('k1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
    });

    it('searchKnowledge calls GET /knowledge/search', async () => {
      const mockData = [{ id: 'k1', content: 'test', relevance: 0.95 }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.searchKnowledge('machine learning');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/knowledge/search');
    });
  });

  describe('Datasets', () => {
    it('listDatasets calls GET /datasets', async () => {
      const mockData = [{ id: 'd1', name: 'train' }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.listDatasets();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/datasets');
    });

    it('getDataset calls GET /datasets/{id}', async () => {
      const mockData = { id: 'd1', name: 'train', rows: 100 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getDataset('d1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/datasets/d1');
    });

    it('getDatasetStats calls GET /datasets/{id}/stats', async () => {
      const mockData = { rows: 100, columns: 5, size_bytes: 1024 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getDatasetStats('d1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/datasets/d1/stats');
    });
  });

  describe('Experiments', () => {
    it('listExperiments calls GET /experiments', async () => {
      const mockData = [{ experiment_id: 'exp1', name: 'test' }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.listExperiments();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/experiments');
    });

    it('getExperiment calls GET /experiments/{id}', async () => {
      const mockData = { experiment_id: 'exp1', name: 'test' };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getExperiment('exp1');

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/experiments/exp1');
    });

    it('logMetric calls POST /experiments/{id}/log_metric', async () => {
      mockFetch.mockResolvedValue(createMockResponse(undefined));

      const client = new SloughGPTClient();
      await client.logMetric('exp1', 'accuracy', 0.95);

      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
      expect(mockFetch.mock.calls[0][0]).toContain('/experiments/exp1/log_metric');
    });
  });

  describe('Audit Log', () => {
    it('getAuditLog calls GET /security/audit', async () => {
      const mockData = [{ action: 'login', timestamp: '2026-09-10' }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getAuditLog();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/security/audit');
    });
  });

  describe('Registry', () => {
    it('listRegistryModels calls GET /registry/models', async () => {
      const mockData = [{ model_id: 'm1' }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.listRegistryModels();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/registry/models');
    });

    it('getRegistryBest calls GET /registry/best', async () => {
      const mockData = { model_id: 'best-m', score: 0.99 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getRegistryBest();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/registry/best');
    });

    it('getRegistryStats calls GET /registry/stats', async () => {
      const mockData = { total_models: 5, avg_score: 0.85 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getRegistryStats();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/registry/stats');
    });
  });

  describe('Benchmark', () => {
    it('getBenchmarkMetrics calls GET /benchmark/metrics', async () => {
      const mockData = [{ name: 'latency', value: 50 }];
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getBenchmarkMetrics();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/benchmark/metrics');
    });

    it('getBenchmarkStats calls GET /benchmark/stats', async () => {
      const mockData = { total_runs: 10, avg_latency: 45 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getBenchmarkStats();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/benchmark/stats');
    });
  });

  describe('Tokenizer', () => {
    it('getTokenizerStats calls GET /tokenizer/stats', async () => {
      const mockData = { vocab_size: 50257, num_tokens: 1000 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getTokenizerStats();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/tokenizer/stats');
    });
  });

  describe('Feedback', () => {
    it('recordFeedback calls POST /feedback/workflow-record', async () => {
      const mockData = { recorded: true };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.recordFeedback({ session_id: 's1', message_id: 'm1', score: 5, tags: ['helpful'] });

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    });

    it('getWorkflowStatus calls GET /workflow/status', async () => {
      const mockData = { active: true, pending: 0 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.getWorkflowStatus();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/workflow/status');
    });
  });

  describe('Metrics', () => {
    it('metrics calls GET /metrics', async () => {
      const mockData = { cpu_percent: 50, memory_percent: 60, disk_percent: 30, uptime_seconds: 1000 };
      mockFetch.mockResolvedValue(createMockResponse(mockData));

      const client = new SloughGPTClient();
      const result = await client.metrics();

      expect(result).toEqual(mockData);
      expect(mockFetch.mock.calls[0][0]).toContain('/metrics');
    });
  });
});
