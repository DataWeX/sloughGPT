import React, {useEffect, useRef, useState} from 'react';
import {Animated} from 'react-native';
import {XStack, YStack, Text} from 'tamagui';

interface Props {
  visible: boolean;
  onDone?: () => void;
}

function Dot({delay}: {delay: number}) {
  const bounce = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.delay(delay),
        Animated.timing(bounce, {toValue: -5, duration: 200, useNativeDriver: true}),
        Animated.timing(bounce, {toValue: 0, duration: 200, useNativeDriver: true}),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, []);

  return (
    <Animated.View
      style={{
        width: 6,
        height: 6,
        borderRadius: 3,
        backgroundColor: '$color9',
        transform: [{translateY: bounce}],
      }}
    />
  );
}

export function ReasoningPanel({visible, onDone}: Props) {
  const [elapsed, setElapsed] = useState(0);
  const [doneVisible, setDoneVisible] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startRef = useRef<number>(0);

  useEffect(() => {
    if (visible) {
      startRef.current = Date.now();
      setElapsed(0);
      setDoneVisible(false);
      timerRef.current = setInterval(() => {
        setElapsed(Date.now() - startRef.current);
      }, 100);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [visible]);

  useEffect(() => {
    if (!visible && elapsed > 0 && !doneVisible) {
      setDoneVisible(true);
      const t = setTimeout(() => {
        setDoneVisible(false);
        onDone?.();
      }, 1500);
      return () => clearTimeout(t);
    }
  }, [visible, elapsed]);

  if (!visible && !doneVisible) return null;

  const elapsedStr = (elapsed / 1000).toFixed(1);

  if (doneVisible) {
    return (
      <XStack paddingHorizontal={16} marginBottom={8} alignItems="flex-start">
        <YStack
          paddingHorizontal={12}
          paddingVertical={8}
          borderRadius={12}
          borderBottomLeftRadius={4}
          backgroundColor="#E8F5EE">
          <Text fontSize={11} fontWeight="500" letterSpacing={0.2} color="#22C55E">
            Reasoning complete ({elapsedStr}s)
          </Text>
        </YStack>
      </XStack>
    );
  }

  return (
    <XStack paddingHorizontal={16} marginBottom={8} alignItems="flex-start">
      <XStack
        alignItems="center"
        backgroundColor="#EEEDFF"
        paddingHorizontal={12}
        paddingVertical={8}
        borderRadius={12}
        borderBottomLeftRadius={4}
        borderWidth={1}
        borderColor="#DDD6FE"
        gap={8}>
        <Text fontSize={11} fontWeight="600" letterSpacing={0.2} color="$color9">
          Reasoning
        </Text>
        <XStack alignItems="center" gap={4}>
          <Dot delay={0} />
          <Dot delay={150} />
          <Dot delay={300} />
        </XStack>
        <Text fontSize={11} color="$color10" fontVariant={['tabular-nums']}>
          {elapsedStr}s
        </Text>
      </XStack>
    </XStack>
  );
}
