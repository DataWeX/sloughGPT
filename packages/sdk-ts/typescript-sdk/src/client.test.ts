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
    it('sends POST /chat and maps { text } to ChatResult', async () => {
      mockFetch.mockResolvedValue(
        createMockResponse({ text: 'Hello!', model: 'gpt2-engine', tokens_generated: 3 })
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
        'http://localhost:8000/souls/switch/friendly',
        expect.objectContaining({ method: 'POST', body: JSON.stringify({}) })
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
        'http://localhost:8000/souls/switch/friendly',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ checkpoint_name: 'ckpt-v2' }),
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
      expect(mockFetch.mock.calls[0][0]).toContain('/auto-train/start');
    });

    it('stops auto-train', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'stopped' }));
      const client = new SloughGPTClient();
      await client.stopAutoTrain();
      expect(mockFetch.mock.calls[0][0]).toContain('/auto-train/stop');
    });

    it('gets auto-train status', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ phase: 'TRAIN' }));
      const client = new SloughGPTClient();
      const result = await client.getAutoTrainStatus();
      expect(result.phase).toBe('TRAIN');
      expect(mockFetch.mock.calls[0][0]).toContain('/auto-train/status');
    });

    it('lists checkpoints', async () => {
      mockFetch.mockResolvedValue(createMockResponse([{ name: 'ckpt-1' }]));
      const client = new SloughGPTClient();
      const result = await client.listAutoTrainCheckpoints();
      expect(result).toHaveLength(1);
      expect(mockFetch.mock.calls[0][0]).toContain('/auto-train/checkpoints');
    });

    it('deletes checkpoint', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'deleted' }));
      const client = new SloughGPTClient();
      await client.deleteAutoTrainCheckpoint('ckpt-1');
      expect(mockFetch.mock.calls[0][0]).toContain('/auto-train/checkpoints/ckpt-1');
      expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
    });

    it('loads checkpoint', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ status: 'loaded' }));
      const client = new SloughGPTClient();
      await client.loadAutoTrainCheckpoint('ckpt-1');
      expect(mockFetch.mock.calls[0][0]).toContain('/auto-train/checkpoints/ckpt-1/load');
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
      mockFetch.mockResolvedValue(createMockResponse({ status: 'success', data: { keys: mockKeys, count: 1 } }));
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
    it('returns async generator with raw SSE tokens', async () => {
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode('data: Hello\n\n'));
          controller.enqueue(encoder.encode('data:  world\n\n'));
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
      expect(tokens).toEqual(['Hello', 'world']);
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
});
