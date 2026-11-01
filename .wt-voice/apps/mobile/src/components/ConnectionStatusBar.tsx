/**
 * Connection status bar — thin strip at top of screen showing server connection state.
 * Animated transitions between states. Tap to retry when offline.
 * Shows latency when connected, retry count when reconnecting.
 */

import React from 'react';
import {Pressable, ActivityIndicator} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {useSafeAreaInsets} from 'react-native-safe-area-context';
import {useConnectionStatus, type ConnectionState} from '../hooks/useConnectionStatus';
import {useColors} from '../theme/colors';

interface Props {
  onRetry?: () => void;
}

function getStateConfig(state: ConnectionState, c: ReturnType<typeof useColors>) {
  switch (state) {
    case 'connected':
      return {bg: c.success, label: 'Connected'};
    case 'connecting':
      return {bg: c.warning, label: 'Connecting...'};
    case 'reconnecting':
      return {bg: c.warning, label: 'Reconnecting...'};
    case 'offline':
      return {bg: c.error, label: 'Offline'};
  }
}

export function ConnectionStatusBar({onRetry}: Props) {
  const {state, latencyMs, retryCount} = useConnectionStatus();
  const insets = useSafeAreaInsets();
  const c = useColors();

  // Hide during initial connecting state (first poll hasn't completed yet)
  if (state === 'connecting' && retryCount === 0) {
    return null;
  }

  const config = getStateConfig(state, c);

  const bar = (
    <YStack backgroundColor={config.bg} paddingHorizontal={12} paddingVertical={6} paddingTop={insets.top || 6}>
      <XStack alignItems="center" gap={6}>
        {state === 'connecting' || state === 'reconnecting' ? (
          <ActivityIndicator size="small" color="#fff" style={{width: 12, height: 12}} />
        ) : (
          <YStack width={6} height={6} borderRadius={3} backgroundColor={c.whiteAlpha(0.8)} />
        )}
        <Text fontSize={11} fontWeight="600" letterSpacing={0.3} color="white">
          {state === 'connected'
            ? latencyMs !== null ? `${latencyMs}ms` : 'Connected'
            : `${config.label}${retryCount > 0 && state !== 'connecting' ? ` (${retryCount})` : ''}`}
        </Text>
        {state === 'offline' && (
          <Text fontSize={10} opacity={0.8} color="white" marginLeft="auto">
            Tap to retry
          </Text>
        )}
      </XStack>
    </YStack>
  );

  if (state === 'offline') {
    return <Pressable onPress={onRetry}>{bar}</Pressable>;
  }

  return bar;
}
