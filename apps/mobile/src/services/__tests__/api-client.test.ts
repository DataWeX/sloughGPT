import AsyncStorage from '@react-native-async-storage/async-storage';
import {api, ApiError, getApiUrl, setApiUrl} from '../api-client';

let mockFetch: jest.Mock;

beforeEach(async () => {
  mockFetch = jest.fn();
  jest.spyOn(global, 'fetch').mockImplementation(mockFetch);
  // Clear persisted API URL between tests
  await AsyncStorage.clear();
});

afterEach(() => {
  jest.restoreAllMocks();
});

function mockResponse(data: unknown, status = 200, statusText = 'OK') {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    statusText,
    text: () => Promise.resolve(JSON.stringify(data)),
    json: () => Promise.resolve(data),
  } as Response);
}

function mockErrorResponse(status: number, detail: string) {
  return Promise.resolve({
    ok: false,
    status,
    statusText: 'Error',
    text: () => Promise.resolve(JSON.stringify({detail})),
    json: () => Promise.resolve({detail}),
  } as Response);
}

describe('getApiUrl / setApiUrl', () => {
  it('returns default url when nothing stored', async () => {
    const url = await getApiUrl();
    expect(url).toBe('http://localhost:8000');
  });

  it('returns stored url after set', async () => {
    await setApiUrl('http://my-server:9000');
    const url = await getApiUrl();
    expect(url).toBe('http://my-server:9000');
  });
});

describe('request', () => {
  it('GET returns parsed data', async () => {
    mockFetch.mockResolvedValue(mockResponse({message: 'ok'}));
    const result = await api.get<{message: string}>('/test');
    expect(result).toEqual({message: 'ok'});
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/test',
      expect.objectContaining({method: 'GET'}),
    );
  });

  it('POST sends body as JSON', async () => {
    mockFetch.mockResolvedValue(mockResponse({id: 1}));
    const result = await api.post<{id: number}>('/test', {name: 'foo'});
    expect(result).toEqual({id: 1});
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/test',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({name: 'foo'}),
      }),
    );
  });

  it('PUT sends body', async () => {
    mockFetch.mockResolvedValue(mockResponse({ok: true}));
    const result = await api.put('/test', {title: 'new'});
    expect(result).toEqual({ok: true});
  });

  it('PATCH sends body', async () => {
    mockFetch.mockResolvedValue(mockResponse({ok: true}));
    const result = await api.patch('/test', {field: 'val'});
    expect(result).toEqual({ok: true});
  });

  it('DELETE sends DELETE method', async () => {
    mockFetch.mockResolvedValue(mockResponse({deleted: true}));
    const result = await api.delete('/test/1');
    expect(result).toEqual({deleted: true});
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/test/1',
      expect.objectContaining({method: 'DELETE'}),
    );
  });

  it('returns undefined for empty response body', async () => {
    mockFetch.mockResolvedValue(
      Promise.resolve({
        ok: true,
        status: 204,
        statusText: 'No Content',
        text: () => Promise.resolve(''),
        json: () => Promise.reject(new Error('no body')),
      } as Response),
    );
    const result = await api.get('/empty');
    expect(result).toBeUndefined();
  });

  it('throws ApiError on 4xx', async () => {
    mockFetch.mockResolvedValue(mockErrorResponse(404, 'Not found'));
    await expect(api.get('/missing')).rejects.toThrow(ApiError);
    await expect(api.get('/missing')).rejects.toThrow('Not found');
  });

  it('throws ApiError with status code', async () => {
    mockFetch.mockResolvedValue(mockErrorResponse(422, 'Validation failed'));
    try {
      await api.get('/bad');
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      expect((e as ApiError).status).toBe(422);
    }
  });

  it('retries once on 5xx error', async () => {
    mockFetch
      .mockResolvedValueOnce(mockErrorResponse(500, 'Server error'))
      .mockResolvedValueOnce(mockResponse({ok: true}));
    const result = await api.get('/retry');
    expect(result).toEqual({ok: true});
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it('retries on network error', async () => {
    mockFetch
      .mockRejectedValueOnce(new Error('Network failure'))
      .mockResolvedValueOnce(mockResponse({ok: true}));
    const result = await api.get('/net-retry');
    expect(result).toEqual({ok: true});
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it('throws after exhausting retries on 5xx', async () => {
    mockFetch
      .mockResolvedValueOnce(mockErrorResponse(500, 'Error'))
      .mockResolvedValueOnce(mockErrorResponse(500, 'Error'));
    await expect(api.get('/fail')).rejects.toThrow('Error');
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it('does not retry on 4xx', async () => {
    mockFetch.mockResolvedValue(mockErrorResponse(400, 'Bad request'));
    await expect(api.get('/bad')).rejects.toThrow('Bad request');
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});

describe('upload', () => {
  it('sends form data and returns result', async () => {
    mockFetch.mockResolvedValue(mockResponse({status: 'uploaded'}));
    const formData = new FormData();
    formData.append('file', 'test' as any);
    const result = await api.upload<{status: string}>('/upload', formData);
    expect(result).toEqual({status: 'uploaded'});
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/upload',
      expect.objectContaining({
        method: 'POST',
        body: formData,
      }),
    );
  });

  it('throws ApiError on failed upload', async () => {
    mockFetch.mockResolvedValue(mockErrorResponse(413, 'Too large'));
    const formData = new FormData();
    formData.append('file', 'big' as any);
    await expect(api.upload('/upload', formData)).rejects.toThrow(ApiError);
  });
});

describe('sync', () => {
  it('POSTs to /mobile/sync', async () => {
    mockFetch.mockResolvedValue(mockResponse({synced: true}));
    const result = await api.sync({pending_messages: [{text: 'hi'}]});
    expect(result).toEqual({synced: true});
  });
});

describe('renameSession', () => {
  it('PUTs new name', async () => {
    mockFetch.mockResolvedValue(mockResponse({ok: true}));
    await api.renameSession('abc', 'New Title');
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/chat/sessions/abc',
      expect.objectContaining({
        method: 'PUT',
        body: JSON.stringify({name: 'New Title'}),
      }),
    );
  });
});

describe('archiveSession', () => {
  it('PUTs archived flag', async () => {
    mockFetch.mockResolvedValue(mockResponse({ok: true}));
    await api.archiveSession('abc', true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/chat/sessions/abc',
      expect.objectContaining({
        method: 'PUT',
        body: JSON.stringify({archived: true}),
      }),
    );
  });
});

describe('searchSessions', () => {
  it('GETs search with query param', async () => {
    mockFetch.mockResolvedValue(mockResponse({results: [{id: '1', text: 'hi'}]}));
    const result = await api.searchSessions('hello');
    expect(result.results).toHaveLength(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/chat/sessions/search?q=hello',
      expect.anything(),
    );
  });

  it('includes limit param when provided', async () => {
    mockFetch.mockResolvedValue(mockResponse({results: []}));
    await api.searchSessions('test', 5);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/chat/sessions/search?q=test&limit=5',
      expect.anything(),
    );
  });
});

describe('sendVoiceMessage', () => {
  it('creates form data and POSTs', async () => {
    mockFetch.mockResolvedValue(mockResponse({
      status: 'ok', message_id: 'm1', audio_path: '/a.m4a', session_id: 's1',
    }));
    const result = await api.sendVoiceMessage('s1', 'file:///a.m4a', 3000);
    expect(result.status).toBe('ok');
    expect(result.session_id).toBe('s1');
  });
});

describe('syncStatus', () => {
  it('GETs sync status', async () => {
    mockFetch.mockResolvedValue(mockResponse({connected: true}));
    const result = await api.syncStatus();
    expect(result).toEqual({connected: true});
  });
});
