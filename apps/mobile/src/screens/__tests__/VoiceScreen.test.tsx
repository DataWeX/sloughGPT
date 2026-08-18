import React from 'react';
import {render, waitFor, fireEvent} from '@/test-utils';

jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {
    SafeAreaView: ({children, style}: any) =>
      React.createElement(View, {style, testID: 'safe-area-view'}, children),
    useSafeAreaInsets: () => ({top: 0, bottom: 0, left: 0, right: 0}),
  };
});

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn().mockResolvedValue(null),
    post: jest.fn().mockResolvedValue({audio_path: '/tmp/test.wav'}),
  },
}));

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}));

jest.mock('../../services/toast', () => ({
  toast: {success: jest.fn(), error: jest.fn(), info: jest.fn()},
}));

jest.mock('../../components/StatusBadge', () => ({
  StatusBadge: ({label}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, null, label);
  },
}));

jest.mock('../../components/Icon', () => ({
  Icon: ({name}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, null, name);
  },
}));

beforeEach(() => {
  jest.clearAllMocks();
});

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {VoiceScreen} = require('../VoiceScreen');

describe('VoiceScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<VoiceScreen />);
    await waitFor(() => {
      expect(getByText('Voice')).toBeTruthy();
    });
  });

  it('shows TTS status section', async () => {
    const {getByText} = await render(<VoiceScreen />);
    await waitFor(() => {
      expect(getByText('TTS Status')).toBeTruthy();
    });
  });

  it('shows test TTS section', async () => {
    const {getByText} = await render(<VoiceScreen />);
    await waitFor(() => {
      expect(getByText('Test TTS')).toBeTruthy();
    });
  });

  it('shows about section', async () => {
    const {getByText} = await render(<VoiceScreen />);
    await waitFor(() => {
      expect(getByText('About')).toBeTruthy();
    });
  });

  it('shows available badge when TTS is available', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce({
      tts_available: true,
      model_name: 'piper-tts',
      tts_calls: 10,
      fallback: 'espeak',
      error: null,
    });
    const {getByText} = await render(<VoiceScreen />);
    await waitFor(() => {
      expect(getByText('piper-tts')).toBeTruthy();
      expect(getByText('10')).toBeTruthy();
    });
  });

  it('shows error when TTS has error', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce({
      tts_available: false,
      model_name: null,
      tts_calls: 0,
      fallback: 'espeak',
      error: 'Model not loaded',
    });
    const {getByText} = await render(<VoiceScreen />);
    await waitFor(() => {
      expect(getByText('Model not loaded')).toBeTruthy();
    });
  });

  it('calls TTS on generate press', async () => {
    const mockApi = require('../../services/api-client').api;
    mockApi.get.mockResolvedValueOnce(null);
    const {getByText, getByPlaceholderText} = await render(<VoiceScreen />);
    await waitFor(() => {
      expect(getByText('Voice')).toBeTruthy();
    });
    const input = getByPlaceholderText('Enter text to synthesize...');
    await waitFor(() => {
      fireEvent.changeText(input, 'Hello world');
    });
    await waitFor(() => {
      fireEvent.press(getByText('Generate & Play'));
    });
    await waitFor(() => {
      expect(mockApi.post).toHaveBeenCalledWith('/voice/tts', {text: 'Hello world'});
    });
  });
});
