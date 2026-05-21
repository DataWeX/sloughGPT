import { SloughGPTClient } from '@sloughgpt/typescript-sdk';
import React, { useCallback, useEffect, useState } from 'react';
import { Box, Text, useApp, useInput, Spacer, Newline } from 'ink';

type Props = {
  baseUrl: string;
};

type View = 'home' | 'local' | 'api' | 'train' | 'docker';

interface LocalStatus {
  models: number;
  datasets: number;
  modelPaths: string[];
  datasetNames: string[];
}

export default function App({ baseUrl }: Props) {
  const { exit } = useApp();
  const [view, setView] = useState<View>('home');
  const [health, setHealth] = useState<{ status: string; loaded: boolean } | null>(null);
  const [localStatus, setLocalStatus] = useState<LocalStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const client = new SloughGPTClient({ baseUrl });

  const fetchHealth = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const h = await client.health();
      setHealth({ status: h.status, loaded: h.model_loaded });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
    setLoading(false);
  }, [client]);

  useEffect(() => {
    void fetchHealth();
  }, [fetchHealth]);

  useInput((input, key) => {
    if (input === 'q' || key.escape) {
      exit();
    }
    if (input === '1') setView('home');
    if (input === '2') setView('local');
    if (input === '3') setView('api');
    if (input === '4') setView('train');
    if (input === '5') setView('docker');
    if (input === 'r') void fetchHealth();
  });

  return (
    <Box flexDirection="column" padding={1}>
      <Box flexDirection="column" borderStyle="round" borderColor="cyan" padding={1}>
        <Text bold color="cyan"> SloughGPT TUI </Text>
        <Text dimColor>API {baseUrl}</Text>
      </Box>

      <Newline />

      <Box>
        <Box flexDirection="column" width={20}>
          <Text bold color="white"> Menu </Text>
          <Text color={view === 'home' ? 'cyan' : 'dim'}>  [1] Home</Text>
          <Text color={view === 'local' ? 'cyan' : 'dim'}>  [2] Local Status</Text>
          <Text color={view === 'api' ? 'cyan' : 'dim'}>  [3] API Health</Text>
          <Text color={view === 'train' ? 'cyan' : 'dim'}>  [4] Training</Text>
          <Text color={view === 'docker' ? 'cyan' : 'dim'}>  [5] Docker</Text>
        </Box>

        <Box flexDirection="column" borderStyle="round" borderColor="white" padding={1} width={50}>
          {view === 'home' && (
            <>
              <Text bold color="cyan"> Welcome </Text>
              <Newline />
              <Text> sloughgpt-tui - Interactive terminal UI </Text>
              <Text dimColor> Use [1-5] to navigate, [r] to refresh, [q] to quit </Text>
            </>
          )}

          {view === 'local' && (
            <>
              <Text bold color="cyan"> Local Status </Text>
              <Newline />
              <Text>Scanning repository...</Text>
              <Text dimColor>(Python TUI provides local status)</Text>
            </>
          )}

          {view === 'api' && (
            <>
              <Text bold color="cyan"> API Health </Text>
              <Newline />
              {loading ? (
                <Text>Loading...</Text>
              ) : error ? (
                <Text color="red">Error: {error}</Text>
              ) : health ? (
                <Box flexDirection="column">
                  <Text> Status: {health.status} </Text>
                  <Text> Model loaded: {String(health.loaded)} </Text>
                </Box>
              ) : null}
            </>
          )}

          {view === 'train' && (
            <>
              <Text bold color="cyan"> Training </Text>
              <Newline />
              <Text dimColor>Use: sloughgpt-tui --train --epochs 3</Text>
              <Text dimColor>(Python TUI provides training)</Text>
            </>
          )}

          {view === 'docker' && (
            <>
              <Text bold color="cyan"> Docker </Text>
              <Newline />
              <Text dimColor>Use: sloughgpt-tui --docker-status</Text>
              <Text dimColor>(Python TUI provides Docker ops)</Text>
            </>
          )}
        </Box>
      </Box>

      <Newline />
      <Text dimColor>[r] refresh  [1-5] navigate  [q] quit</Text>
    </Box>
  );
}