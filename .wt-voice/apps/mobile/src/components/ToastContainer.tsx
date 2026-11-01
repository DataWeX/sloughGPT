/**
 * Toast notification container — renders toast stack at top of screen.
 * Animates in/out. Supports success/error/info/warn types.
 * Import and render once at app root level.
 */

import React, {useEffect, useState} from 'react';
import {Pressable, Animated} from 'react-native';
import {YStack, Text} from 'tamagui';
import {toast, type Toast, type ToastType} from '../services/toast';
import {Icon, type IconName} from './Icon';
import {useColors} from '../theme/colors';

function getTypeStyle(type: ToastType, c: ReturnType<typeof useColors>) {
  switch (type) {
    case 'success': return {bg: c.successLight, border: c.success, text: c.successDark};
    case 'error':   return {bg: c.errorLight, border: c.error, text: c.errorDark};
    case 'info':    return {bg: c.infoLight, border: c.info, text: c.infoDark};
    case 'warn':    return {bg: c.warningLight, border: c.warning, text: c.warningDark};
  }
}

const ICON_NAMES: Record<ToastType, IconName> = {
  success: 'check',
  error: 'x',
  info: 'info',
  warn: 'triangle-alert',
};

function ToastItem({item}: {item: Toast}) {
  const [opacity] = useState(new Animated.Value(0));
  const c = useColors();
  const colors = getTypeStyle(item.type, c);

  useEffect(() => {
    Animated.timing(opacity, {
      toValue: 1,
      duration: 200,
      useNativeDriver: true,
    }).start();

    return () => {
      Animated.timing(opacity, {
        toValue: 0,
        duration: 150,
        useNativeDriver: true,
      }).start();
    };
  }, []);

  return (
    <Animated.View
      style={[
        {
          flexDirection: 'row',
          alignItems: 'center',
          paddingHorizontal: 12,
          paddingVertical: 10,
          borderRadius: 10,
          borderLeftWidth: 3,
          backgroundColor: colors.bg,
          borderLeftColor: colors.border,
          opacity,
        },
      ]}>
      <YStack width={20} height={20} borderRadius={10} backgroundColor={colors.border} alignItems="center" justifyContent="center" marginRight={8}>
        <Icon name={ICON_NAMES[item.type]} size={11} color={c.white} />
      </YStack>
      <Text fontSize={13} lineHeight={18} color={colors.text} flex={1} numberOfLines={2}>
        {item.message}
      </Text>
      <Pressable onPress={() => toast.dismiss(item.id)} style={{padding: 4, marginLeft: 8}}>
        <Icon name="x" size={16} color={colors.text} />
      </Pressable>
    </Animated.View>
  );
}

export function ToastContainer() {
  const [items, setItems] = useState<Toast[]>([]);

  useEffect(() => {
    setItems(toast.getToasts());
    return toast.subscribe(setItems);
  }, []);

  if (items.length === 0) return null;

  return (
    <YStack position="absolute" top={44} left={12} right={12} zIndex={99999} gap={6} pointerEvents="box-none">
      {items.map(item => (
        <ToastItem key={item.id} item={item} />
      ))}
    </YStack>
  );
}
