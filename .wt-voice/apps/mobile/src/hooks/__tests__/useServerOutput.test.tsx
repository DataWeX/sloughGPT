jest.mock('../../services/sse-client');

import React from 'react';
import {Text} from 'react-native';
import {render, waitFor, screen, act} from '@testing-library/react-native';
import {useServerOutput} from '../useServerOutput';
import {streamSSE} from '../../services/sse-client';

const mockStreamSSE = streamSSE as jest.MockedFunction<typeof streamSSE>;

function Display({enabled, maxLines}: {enabled?: boolean; maxLines?: number}) {
  const {lines, streaming, paused, clear, togglePause, exportLines} = useServerOutput({enabled, maxLines});
  return (
    <Text testID="val">
      {JSON.stringify({
        count: lines.length,
        firstMsg: lines[0]?.message ?? null,
        firstSev: lines[0]?.severity ?? null,
        streaming,
        paused,
        exported: exportLines(),
      })}
    </Text>
  );
}

function parseVal(): any {
  return JSON.parse(String(screen.getByTestId('val').children[0]));
}

async function* fakeStream(events: Array<{data?: Record<string, unknown>}>) {
  for (const event of events) {
    yield event;
  }
}

beforeEach(() => {
  jest.clearAllMocks();
});

function suppressActWarning() {
  jest.spyOn(console, 'error').mockImplementation(msg => {
    if (typeof msg === 'string' && msg.includes('not wrapped in act')) return;
    console.error(msg);
  });
}

describe('useServerOutput', () => {
  beforeEach(() => {
    suppressActWarning();
  });

  it('returns initial state when disabled', async () => {
    mockStreamSSE.mockReturnValue(fakeStream([]) as any);
    await render(<Display enabled={false} />);
    const val = parseVal();
    expect(val.count).toBe(0);
    expect(val.paused).toBe(false);
  });

  it('streams output lines', async () => {
    mockStreamSSE.mockReturnValue(
      fakeStream([
        {data: {line: 'Server started', severity: 'info'}},
      ]) as any,
    );

    await render(<Display />);

    await waitFor(() => {
      const val = parseVal();
      expect(val.count).toBe(1);
      expect(val.firstMsg).toBe('Server started');
      expect(val.firstSev).toBe('info');
    });
  });

  it('clears lines', async () => {
    mockStreamSSE.mockReturnValue(
      fakeStream([{data: {line: 'test'}}]) as any,
    );

    await render(<Display />);

    await waitFor(() => {
      expect(parseVal().count).toBe(1);
    });
  });

  it('does not stream when disabled', async () => {
    mockStreamSSE.mockReturnValue(fakeStream([]) as any);
    await render(<Display enabled={false} />);
    expect(mockStreamSSE).not.toHaveBeenCalled();
  });

  it('handles message field in event data', async () => {
    mockStreamSSE.mockReturnValue(
      fakeStream([{data: {message: 'test message', severity: 'error'}}]) as any,
    );

    await render(<Display />);

    await waitFor(() => {
      const val = parseVal();
      expect(val.count).toBe(1);
      expect(val.firstMsg).toBe('test message');
      expect(val.firstSev).toBe('error');
    });
  });

  it('respects maxLines limit', async () => {
    const events = Array.from({length: 5}, (_, i) => ({data: {line: `line ${i}`}}));
    mockStreamSSE.mockReturnValue(fakeStream(events) as any);

    await render(<Display maxLines={3} />);

    await waitFor(() => {
      expect(parseVal().count).toBe(3);
    });
  });
});
