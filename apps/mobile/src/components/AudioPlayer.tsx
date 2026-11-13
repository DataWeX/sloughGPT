import React, {useState, useRef, useEffect, useCallback} from 'react';
import {Pressable, Animated} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {getApiUrl} from '../services/api-client';

interface Props {
  audioUrl?: string;
  audioPath?: string;
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
        soundRef.current.unloadAsync().catch((e: unknown) => {
          if (__DEV__) console.warn('[AudioPlayer] unload failed:', e);
        });
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
        } catch (e) {
          if (__DEV__) console.warn('[AudioPlayer] position poll error:', e);
        }
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
    } catch (e) {
      if (__DEV__) console.warn('[AudioPlayer] playback error:', e);
    }
  }, [playing, resolvedUrl, startPositionPolling]);

  const progressWidth = animWidth.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  if (!resolvedUrl) {
    return (
      <XStack backgroundColor="$background" borderRadius={8} padding={8} alignItems="center" minWidth={180} marginTop={4}>
        <Text fontSize={11} fontWeight="500" letterSpacing={0.2} color="$color10" fontStyle="italic" marginLeft={8}>
          voice
        </Text>
      </XStack>
    );
  }

  return (
    <XStack
      backgroundColor="$background"
      borderRadius={8}
      padding={8}
      alignItems="center"
      minWidth={180}
      marginTop={4}>
      <Pressable onPress={togglePlay}>
        <YStack
          width={32}
          height={32}
          borderRadius={16}
          backgroundColor="$color9"
          alignItems="center"
          justifyContent="center"
          marginRight={8}>
          <Text fontSize={14} color="white">
            {playing ? '\u23F8' : '\u25B6'}
          </Text>
        </YStack>
      </Pressable>
      <XStack flex={1} alignItems="center">
        <YStack flex={1} height={4} backgroundColor="$borderColor" borderRadius={2} overflow="hidden">
          <Animated.View
            style={{
              height: '100%',
              backgroundColor: '$color9',
              borderRadius: 2,
              width: progressWidth,
            }}
          />
        </YStack>
        <Text
          fontSize={11}
          fontWeight="500"
          letterSpacing={0.2}
          color="$color10"
          marginLeft={8}
          minWidth={32}
          textAlign="right">
          {playing ? formatDuration(position) : durationMs > 0 ? formatDuration(durationMs) : ''}
        </Text>
      </XStack>
      {!Audio && resolvedUrl && (
        <Text fontSize={11} fontWeight="500" letterSpacing={0.2} color="$color10" fontStyle="italic" marginLeft={8}>
          audio
        </Text>
      )}
    </XStack>
  );
}
