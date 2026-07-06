import React, {useState, useRef, useEffect, useCallback} from 'react';
import {View, Text, TouchableOpacity, StyleSheet, Animated} from 'react-native';
import {getApiUrl} from '../services/api-client';
import {colors, spacing, radii, typography} from '../theme';

interface Props {
  /** Direct audio URL (base64, file URI, or full HTTP URL). */
  audioUrl?: string;
  /** Server-side audio path (e.g. "session-id/msg-id.m4a"). Resolved via API base. */
  audioPath?: string;
  /** Duration in milliseconds. */
  durationMs?: number;
}

let Audio: any;
try {
  Audio = require('expo-av').Audio;
} catch {
  Audio = null;
}

export function formatDuration(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${sec.toString().padStart(2, '0')}`;
}

export function useResolvedUrl(audioUrl?: string, audioPath?: string) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (audioUrl) {
      setUrl(audioUrl);
    } else if (audioPath) {
      getApiUrl().then(base => {
        setUrl(`${base}/chat/audio/${audioPath}`);
      });
    }
  }, [audioUrl, audioPath]);

  return url;
}

export function AudioPlayer({audioUrl, audioPath, durationMs = 0}: Props) {
  const resolvedUrl = useResolvedUrl(audioUrl, audioPath);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const soundRef = useRef<any>(null);
  const animWidth = useRef(new Animated.Value(0)).current;
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (soundRef.current) {
        soundRef.current.unloadAsync().catch(() => {});
      }
    };
  }, []);

  const startPositionPolling = useCallback(() => {
    intervalRef.current = setInterval(async () => {
      if (soundRef.current) {
        try {
          const status = await soundRef.current.getStatusAsync();
          if (status.isLoaded) {
            setPosition(status.positionMillis);
            const progress = status.durationMillis
              ? status.positionMillis / status.durationMillis
              : 0;
            animWidth.setValue(progress);
            if (status.didJustFinish) {
              setPlaying(false);
              setPosition(0);
              animWidth.setValue(0);
              if (intervalRef.current) clearInterval(intervalRef.current);
            }
          }
        } catch {}
      }
    }, 250);
  }, [animWidth]);

  const togglePlay = useCallback(async () => {
    if (!Audio || !resolvedUrl) return;

    if (playing) {
      await soundRef.current?.pauseAsync();
      setPlaying(false);
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }

    try {
      if (!soundRef.current) {
        const {sound} = await Audio.Sound.createAsync(
          {uri: resolvedUrl},
          {shouldPlay: true},
        );
        soundRef.current = sound;
      } else {
        await soundRef.current.playAsync();
      }
      setPlaying(true);
      startPositionPolling();
    } catch {
      // Playback failed silently
    }
  }, [playing, resolvedUrl, startPositionPolling]);

  const progressWidth = animWidth.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  if (!resolvedUrl) {
    return (
      <View style={styles.container}>
        <Text style={styles.fallbackBadge}>voice</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={togglePlay} style={styles.playBtn} activeOpacity={0.7}>
        <Text style={styles.playIcon}>{playing ? '⏸' : '▶'}</Text>
      </TouchableOpacity>
      <View style={styles.progressContainer}>
        <View style={styles.progressTrack}>
          <Animated.View style={[styles.progressFill, {width: progressWidth}]} />
        </View>
        <Text style={styles.time}>
          {playing ? formatDuration(position) : durationMs > 0 ? formatDuration(durationMs) : ''}
        </Text>
      </View>
      {!Audio && resolvedUrl && <Text style={styles.fallbackBadge}>audio</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    padding: spacing.sm,
    minWidth: 180,
    marginTop: spacing.xs,
  },
  playBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.sm,
  },
  playIcon: {
    fontSize: 14,
    color: '#fff',
  },
  progressContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
  },
  progressTrack: {
    flex: 1,
    height: 4,
    backgroundColor: colors.border,
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: colors.primary,
    borderRadius: 2,
  },
  time: {
    ...typography.small,
    color: colors.textMuted,
    marginLeft: spacing.sm,
    minWidth: 32,
    textAlign: 'right',
  },
  fallbackBadge: {
    ...typography.small,
    color: colors.textMuted,
    marginLeft: spacing.sm,
    fontStyle: 'italic',
  },
});
