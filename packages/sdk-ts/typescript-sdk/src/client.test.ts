import { describe, it, expect, vi, beforeEach } from 'vitest';
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
    it('sends generate request with defaults', async () => {
      const mockResult = { text: 'Hello world', model: 'gpt2', inference_time_ms: 150 };
      mockFetch.mockResolvedValue(createMockResponse(mockResult));

      const client = new SloughGPTClient();
      const result = await client.generate({ prompt: 'Say hello' });

      expect(result).toEqual(mockResult);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/generate',
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

    it('sends generate request with custom params', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ text: 'test', model: 'gpt2' }));

      const client = new SloughGPTClient();
      await client.generate({
        prompt: 'Write a story',
        max_new_tokens: 200,
        temperature: 0.5,
        top_k: 100,
        top_p: 0.95,
        personality: 'pirate',
        model: 'gpt2-medium',
      });

      const call = mockFetch.mock.calls[0];
      const body = JSON.parse((call[1] as { body: string }).body);
      expect(body.max_new_tokens).toBe(200);
      expect(body.temperature).toBe(0.5);
      expect(body.personality).toBe('pirate');
      expect(body.model).toBe('gpt2-medium');
    });
  });

  describe('chat()', () => {
    it('sends POST /chat and maps { text } to ChatResult', async () => {
      mockFetch.mockResolvedValue(
        createMockResponse({
          text: 'Hello!',
          model: 'gpt2-engine',
          tokens_generated: 3,
        })
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
      const call = mockFetch.mock.calls[0];
      expect(call[0]).toContain('/chat');
      expect((call[1] as { method: string }).method).toBe('POST');
      const body = JSON.parse((call[1] as { body: string }).body);
      expect(body.messages).toHaveLength(1);
      expect(body.max_new_tokens).toBe(150);
    });
  });

  describe('quickGenerate()', () => {
    it('returns just the text', async () => {
      mockFetch.mockResolvedValue(createMockResponse({ text: 'Simplified response', model: 'gpt2' }));

      const client = new SloughGPTClient();
      const result = await client.quickGenerate('Hello');

      expect(result).toBe('Simplified response');
    });
  });

  describe('quickChat()', () => {
    it('returns just the message content', async () => {
      mockFetch.mockResolvedValue(
        createMockResponse({
          text: 'Quick reply',
          model: 'gpt2-engine',
        })
      );

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
        (error as unknown as { name: string }).name = 'AbortError';
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

    it('lists experiments', async () => {
      const mockExps = [{ experiment_id: 'exp-1', name: 'Test' }];
      mockFetch.mockResolvedValue(createMockResponse(mockExps));

      const client = new SloughGPTClient();
      const result = await client.listExperiments();

      expect(result).toEqual(mockExps);
    });
  });

  describe('startTraining()', () => {
    it('sends canonical TrainingRequest body', async () => {
      const job = {
        id: 'job_1',
        name: 'run-a',
        model: 'sloughgpt',
        dataset: 'openwebtext',
        status: 'running' as const,
        progress: 0,
      };
      mockFetch.mockResolvedValue(createMockResponse(job));

      const client = new SloughGPTClient();
      const result = await client.startTraining({
        name: 'run-a',
        model: 'sloughgpt',
        dataset: 'openwebtext',
        epochs: 3,
        log_interval: 5,
        eval_interval: 50,
      });

      expect(result).toEqual(job);
      const [, init] = mockFetch.mock.calls[0];
      const body = JSON.parse((init as RequestInit).body as string);
      expect(body).toEqual({
        name: 'run-a',
        model: 'sloughgpt',
        dataset: 'openwebtext',
        epochs: 3,
        log_interval: 5,
        eval_interval: 50,
      });
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/training/start',
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('maps legacy model_name and dataset_id', async () => {
      mockFetch.mockResolvedValue(
        createMockResponse({
          id: 'job_2',
          status: 'running',
          progress: 0,
        })
      );

      const client = new SloughGPTClient();
      await client.startTraining({
        model_name: 'm1',
        dataset_id: 'corpus',
        epochs: 1,
      });

      const [, init] = mockFetch.mock.calls[0];
      const body = JSON.parse((init as RequestInit).body as string);
      expect(body.model).toBe('m1');
      expect(body.dataset).toBe('corpus');
      expect(body.name).toBe('m1-training');
      expect(body.epochs).toBe(1);
      expect(body.model_name).toBeUndefined();
      expect(body.dataset_id).toBeUndefined();
    });

    it('prefers manifest_uri over dataset when provided', async () => {
      mockFetch.mockResolvedValue(
        createMockResponse({ id: 'job_3', status: 'running', progress: 0 })
      );

      const client = new SloughGPTClient();
      await client.startTraining({
        name: 'x',
        model: 'sloughgpt',
        manifest_uri: 'datasets/a/dataset_manifest.json',
      });

      const [, init] = mockFetch.mock.calls[0];
      const body = JSON.parse((init as RequestInit).body as string);
      expect(body.manifest_uri).toBe('datasets/a/dataset_manifest.json');
      expect(body.dataset).toBeUndefined();
    });
  });

  describe('registry', () => {
    it('registers a model', async () => {
      const mockModel = { model_id: 'model-1', name: 'GPT-2' };
      mockFetch.mockResolvedValue(createMockResponse(mockModel));

      const client = new SloughGPTClient();
      const result = await client.registerModel({ name: 'GPT-2', model_type: 'gpt2' });

      expect(result).toEqual(mockModel);
    });

    it('lists registered models', async () => {
      const mockModels = [{ model_id: 'model-1', name: 'GPT-2' }];
      mockFetch.mockResolvedValue(createMockResponse(mockModels));

      const client = new SloughGPTClient();
      const result = await client.listRegisteredModels();

      expect(result).toEqual(mockModels);
    });
  });
});
