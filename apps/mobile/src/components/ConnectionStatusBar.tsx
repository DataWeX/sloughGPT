/**
 * Connection status bar — thin strip at top of screen showing server connection state.
 * Animated transitions between states. Tap to retry when offline.
 * Shows latency when connected, retry count when reconnecting.
 */

import React from 'react';
import {View, Text, TouchableOpacity, StyleSheet, ActivityIndicator} from 'react-native';
import {useSafeAreaInsets} from 'react-native-safe-area-context';
import {useConnectionStatus, type ConnectionState} from '../hooks/useConnectionStatus';

interface Props {
  onRetry?: () => void;
}

const STATE_CONFIG: Record<ConnectionState, {bg: string; text: string; label: string}> = {
  connected: {bg: '#16a34a', text: '#fff', label: 'Connected'},
  connecting: {bg: '#d97706', text: '#fff', label: 'Connecting...'},
  reconnecting: {bg: '#d97706', text: '#fff', label: 'Reconnecting...'},
  offline: {bg: '#dc2626', text: '#fff', label: 'Offline'},
};

export function ConnectionStatusBar({onRetry}: Props) {
  const {state, latencyMs, retryCount} = useConnectionStatus();
  const insets = useSafeAreaInsets();
  const config = STATE_CONFIG[state];

  // Don't show when connected and healthy
  if (state === 'connected') {
    return (
      <View style={[styles.bar, {backgroundColor: config.bg, paddingTop: insets.top || 6}]}>
        <View style={styles.row}>
          <View style={styles.dot} />
          <Text style={[styles.label, {color: config.text}]}>
            {latencyMs !== null ? `${latencyMs}ms` : 'Connected'}
          </Text>
        </View>
      </View>
    );
  }

  return (
    <TouchableOpacity
      style={[styles.bar, {backgroundColor: config.bg, paddingTop: insets.top || 6}]}
      onPress={onRetry}
      activeOpacity={0.7}>
      <View style={styles.row}>
        {state === 'connecting' || state === 'reconnecting' ? (
          <ActivityIndicator size="small" color={config.text} style={styles.spinner} />
        ) : (
          <View style={[styles.dot, {backgroundColor: config.text}]} />
        )}
        <Text style={[styles.label, {color: config.text}]}>
          {config.label}
          {retryCount > 0 && state !== 'connecting' ? ` (${retryCount})` : ''}
        </Text>
        {state === 'offline' && (
          <Text style={[styles.retryText, {color: config.text}]}>Tap to retry</Text>
        )}
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  bar: {
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: 'rgba(255,255,255,0.8)',
  },
  spinner: {
    width: 12,
    height: 12,
  },
  label: {
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 0.3,
  },
  retryText: {
    fontSize: 10,
    opacity: 0.8,
    marginLeft: 'auto',
  },
});
