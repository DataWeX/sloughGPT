/* eslint-disable @typescript-eslint/no-var-requires */

jest.mock('../haptics');
jest.mock('../sounds');
jest.mock('../toast');

const mockFetch = jest.fn();
global.fetch = mockFetch;

const {streamProviderChat, providerChat, ProviderError} = require('../providers-client');
const {PROVIDER_REGISTRY} = require('../../types/providers');

function makeOpenAIStream(tokens: string[]) {
  const encoder = new TextEncoder();
  const chunks = tokens.map(t => {
    const data = JSON.stringify({choices: [{delta: {content: t}, finish_reason: null}]});
    return `data: ${data}\n\n`;
  });
  chunks.push('data: [DONE]\n\n');

  return new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      },
    }),
    {status: 200, headers: {'content-type': 'text/event-stream'}},
  );
}

function makeOpenAIResponse(text: string) {
  return new Response(
    JSON.stringify({
      choices: [{message: {content: text}, finish_reason: 'stop'}],
      usage: {total_tokens: 42},
    }),
    {status: 200, headers: {'content-type': 'application/json'}},
  );
}

function makeErrorResponse(status: number, body: string) {
  return new Response(body, {status, headers: {'content-type': 'application/json'}});
}

const OPENAI_CONFIG = {
  id: 'openai',
  name: 'OpenAI',
  baseUrl: 'https://api.openai.com/v1',
  apiKey: 'sk-test-123',
  defaultModel: 'gpt-4o-mini',
  enabled: true,
};

const ANTHROPIC_CONFIG = {
  id: 'anthropic',
  name: 'Anthropic',
  baseUrl: 'https://api.anthropic.com/v1',
  apiKey: 'sk-ant-test',
  defaultModel: 'claude-3-5-sonnet-20241022',
  enabled: true,
};

const GEMINI_CONFIG = {
  id: 'google',
  name: 'Google Gemini',
  baseUrl: 'https://generativelanguage.googleapis.com/v1beta',
  apiKey: 'ai-test-key',
  defaultModel: 'gemini-2.0-flash',
  enabled: true,
};

beforeEach(() => {
  jest.clearAllMocks();
});

describe('providers-client', () => {
  describe('streamProviderChat (OpenAI-compatible)', () => {
    it('streams tokens from OpenAI', async () => {
      mockFetch.mockResolvedValueOnce(makeOpenAIStream(['Hello', ' world']));
      const onToken = jest.fn();

      const result = await streamProviderChat(OPENAI_CONFIG, [{role: 'user', content: 'Hi'}], {
        onToken,
      });

      expect(result.text).toBe('Hello world');
      expect(onToken).toHaveBeenCalledTimes(2);
      expect(onToken).toHaveBeenCalledWith('Hello');
      expect(onToken).toHaveBeenCalledWith(' world');
    });

    it('sends correct request to /chat/completions', async () => {
      mockFetch.mockResolvedValueOnce(makeOpenAIStream(['ok']));

      await streamProviderChat(OPENAI_CONFIG, [
        {role: 'system', content: 'Be helpful'},
        {role: 'user', content: 'Hi'},
      ]);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.openai.com/v1/chat/completions',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            Authorization: 'Bearer sk-test-123',
          }),
        }),
      );
    });

    it('throws ProviderError on non-200 response', async () => {
      mockFetch.mockResolvedValueOnce(makeErrorResponse(401, JSON.stringify({error: {message: 'Unauthorized'}})));

      await expect(
        streamProviderChat(OPENAI_CONFIG, [{role: 'user', content: 'Hi'}]),
      ).rejects.toThrow(ProviderError);
    });
  });

  describe('providerChat (OpenAI non-streaming)', () => {
    it('returns full response text', async () => {
      mockFetch.mockResolvedValueOnce(makeOpenAIResponse('Hello world'));

      const result = await providerChat(OPENAI_CONFIG, [{role: 'user', content: 'Hi'}]);

      expect(result.text).toBe('Hello world');
      expect(result.tokensUsed).toBe(42);
    });
  });

  describe('Anthropic provider', () => {
    it('sends correct headers', async () => {
      mockFetch.mockResolvedValueOnce(makeOpenAIStream(['Hi']));

      await streamProviderChat(ANTHROPIC_CONFIG, [{role: 'user', content: 'Hello'}]);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.anthropic.com/v1/messages',
        expect.objectContaining({
          headers: expect.objectContaining({
            'x-api-key': 'sk-ant-test',
            'anthropic-version': '2023-06-01',
          }),
        }),
      );
    });
  });

  describe('Google Gemini provider', () => {
    it('sends correct URL with API key', async () => {
      mockFetch.mockResolvedValueOnce(makeOpenAIStream(['Hi']));

      await streamProviderChat(GEMINI_CONFIG, [{role: 'user', content: 'Hello'}]);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('generativelanguage.googleapis.com'),
        expect.anything(),
      );
    });
  });

  describe('ProviderError', () => {
    it('has correct name', () => {
      const err = new ProviderError('openai', 401, 'Unauthorized');
      expect(err.name).toBe('ProviderError');
      expect(err.provider).toBe('openai');
      expect(err.status).toBe(401);
      expect(err.message).toBe('Unauthorized');
    });
  });
});
