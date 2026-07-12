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

interface Props {
  onRetry?: () => void;
}

const STATE_CONFIG: Record<ConnectionState, {bg: string; label: string}> = {
  connected: {bg: '#16a34a', label: 'Connected'},
  connecting: {bg: '#d97706', label: 'Connecting...'},
  reconnecting: {bg: '#d97706', label: 'Reconnecting...'},
  offline: {bg: '#dc2626', label: 'Offline'},
};

export function ConnectionStatusBar({onRetry}: Props) {
  const {state, latencyMs, retryCount} = useConnectionStatus();
  const insets = useSafeAreaInsets();
  const config = STATE_CONFIG[state];

  const bar = (
    <YStack backgroundColor={config.bg} paddingHorizontal={12} paddingVertical={6} paddingTop={insets.top || 6}>
      <XStack alignItems="center" gap={6}>
        {state === 'connecting' || state === 'reconnecting' ? (
          <ActivityIndicator size="small" color="#fff" style={{width: 12, height: 12}} />
        ) : (
          <YStack width={6} height={6} borderRadius={3} backgroundColor="rgba(255,255,255,0.8)" />
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
