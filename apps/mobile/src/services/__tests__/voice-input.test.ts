jest.mock('../../services/api-client', () => ({
  getApiUrl: jest.fn().mockResolvedValue('http://localhost:8000'),
}));

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

jest.mock('react-native', () => ({
  Alert: {alert: jest.fn()},
  Linking: {openSettings: jest.fn().mockResolvedValue(undefined)},
}));

jest.mock('expo-av', () => {
  const mockInstance = {
    prepareToRecordAsync: jest.fn().mockResolvedValue(undefined),
    startAsync: jest.fn().mockResolvedValue(undefined),
    stopAndUnloadAsync: jest.fn().mockResolvedValue(undefined),
    getURI: jest.fn().mockReturnValue('file:///rec.m4a'),
  };
  const MockRecording = jest.fn(() => mockInstance);
  (MockRecording as any).requestPermissionsAsync = jest.fn().mockResolvedValue({granted: true});
  return {
    __esModule: true,
    Recording: MockRecording,
    RecordingOptionsPresets: {HIGH_QUALITY: {}},
  };
});

import {startRecording, transcribeAudio} from '../voice-input';

beforeEach(() => jest.clearAllMocks());

describe('voice-input', () => {
  it('startRecording returns stop function', async () => {
    const result = await startRecording();
    expect(typeof result.stop).toBe('function');
  });

  it('stop returns recording with uri and duration', async () => {
    const {stop} = await startRecording();
    const recording = await stop();
    expect(recording).toEqual({uri: 'file:///rec.m4a', duration: expect.any(Number)});
  });

  it('transcribeAudio returns empty on fetch failure', async () => {
    jest.spyOn(global, 'fetch').mockRejectedValue(new Error('network'));
    const result = await transcribeAudio('file:///bad.m4a');
    expect(result).toBe('');
    jest.restoreAllMocks();
  });

  it('transcribeAudio posts to /multimodal/transcribe', async () => {
    const fetchSpy = jest.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({text: 'hello world'}),
    } as Response);
    const result = await transcribeAudio('file:///rec.m4a');
    expect(result).toBe('hello world');
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining('/multimodal/transcribe'),
      expect.objectContaining({method: 'POST'}),
    );
    fetchSpy.mockRestore();
  });
});
