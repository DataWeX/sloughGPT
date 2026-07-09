/**
 * Toast notification container — renders toast stack at top of screen.
 * Animates in/out. Supports success/error/info/warn types.
 * Import and render once at app root level.
 */

import React, {useEffect, useState} from 'react';
import {View, Text, TouchableOpacity, StyleSheet, Animated} from 'react-native';
import {toast, type Toast, type ToastType} from '../services/toast';
import {Icon, type IconName} from './Icon';

const TYPE_COLORS: Record<ToastType, {bg: string; border: string; text: string}> = {
  success: {bg: '#f0fdf4', border: '#16a34a', text: '#15803d'},
  error: {bg: '#fef2f2', border: '#dc2626', text: '#b91c1c'},
  info: {bg: '#eff6ff', border: '#2563eb', text: '#1d4ed8'},
  warn: {bg: '#fffbeb', border: '#d97706', text: '#b45309'},
};

const ICON_NAMES: Record<ToastType, IconName> = {
  success: 'check',
  error: 'x',
  info: 'info',
  warn: 'triangle-alert',
};

function ToastItem({item}: {item: Toast}) {
  const [opacity] = useState(new Animated.Value(0));
  const colors = TYPE_COLORS[item.type];

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
    <Animated.View style={[styles.toast, {backgroundColor: colors.bg, borderLeftColor: colors.border, opacity}]}>
      <View style={[styles.icon, {backgroundColor: colors.border}]}>
        <Icon name={ICON_NAMES[item.type]} size={11} color="#fff" />
      </View>
      <Text style={[styles.message, {color: colors.text}]} numberOfLines={2}>
        {item.message}
      </Text>
      <TouchableOpacity onPress={() => toast.dismiss(item.id)} style={styles.closeBtn}>
        <Icon name="x" size={16} color={colors.text} />
      </TouchableOpacity>
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
    <View style={styles.container} pointerEvents="box-none">
      {items.map(item => (
        <ToastItem key={item.id} item={item} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 44,
    left: 12,
    right: 12,
    zIndex: 9999,
    gap: 6,
  },
  toast: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 8,
    borderLeftWidth: 3,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  icon: {
    width: 20,
    height: 20,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 8,
  },
  message: {
    flex: 1,
    fontSize: 13,
    lineHeight: 18,
  },
  closeBtn: {
    padding: 4,
    marginLeft: 8,
  },

});
