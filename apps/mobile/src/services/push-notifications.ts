/**
 * Push notification client service.
 * Handles device token registration, notification listening, and topic management.
 * Uses Expo Push Notifications for cross-platform delivery.
 */

import {Platform} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {api} from './api-client';

const PUSH_TOKEN_KEY = '@sloughgpt/push_token';
const NOTIFICATIONS_ENABLED_KEY = '@sloughgpt/notifications_enabled';

let Notifications: any = null;
try {
  Notifications = require('expo-notifications');
} catch {
  // expo-notifications not installed
}

// ── Configuration ───────────────────────────────────────────────────────────

Notifications?.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

// ── Token Management ────────────────────────────────────────────────────────

export async function registerForPushNotifications(): Promise<string | null> {
  if (!Notifications) return null;

  try {
    const {status: existingStatus} = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== 'granted') {
      const {status} = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }

    if (finalStatus !== 'granted') {
      if (__DEV__) console.log('Push notification permission denied');
      return null;
    }

    const tokenData = await Notifications.getExpoPushTokenAsync();
    const token = tokenData.data;

    // Store locally
    await AsyncStorage.setItem(PUSH_TOKEN_KEY, token);
    await AsyncStorage.setItem(NOTIFICATIONS_ENABLED_KEY, 'true');

    // Register with backend
    try {
      await api.post('/mobile/notifications/register', {
        token,
        platform: Platform.OS,
        user_id: 'default',
        topics: ['chat', 'training'],
      });
    } catch {
      // backend unavailable — token stored locally for retry
    }

    return token;
  } catch (err) {
    if (__DEV__) console.warn('[push-notifications] registration failed:', err);
    return null;
  }
}

export async function getStoredPushToken(): Promise<string | null> {
  return AsyncStorage.getItem(PUSH_TOKEN_KEY);
}

export async function unregisterPushNotifications(): Promise<void> {
  const token = await AsyncStorage.getItem(PUSH_TOKEN_KEY);
  if (token) {
    try {
      await api.post('/mobile/notifications/unregister', {token});
    } catch (e) {
      if (__DEV__) console.warn('[push-notifications] unregister failed:', e);
    }
    await AsyncStorage.removeItem(PUSH_TOKEN_KEY);
    await AsyncStorage.setItem(NOTIFICATIONS_ENABLED_KEY, 'false');
  }
}

export async function isNotificationsEnabled(): Promise<boolean> {
  const val = await AsyncStorage.getItem(NOTIFICATIONS_ENABLED_KEY);
  return val === 'true';
}

// ── Listening ───────────────────────────────────────────────────────────────

type NotificationHandler = (title: string, body: string, data: any) => void;

let _subscribers: NotificationHandler[] = [];
let _subscription: any = null;

export function onNotification(handler: NotificationHandler): () => void {
  _subscribers.push(handler);

  if (!_subscription && Notifications) {
    _subscription = Notifications.addNotificationReceivedListener(
      (notification: any) => {
        const {title, body, data} = notification.request.content;
        _subscribers.forEach(h => h(title || '', body || '', data || {}));
      },
    );
  }

  return () => {
    _subscribers = _subscribers.filter(h => h !== handler);
    if (_subscribers.length === 0 && _subscription) {
      _subscription?.remove();
      _subscription = null;
    }
  };
}

export function onNotificationResponse(
  handler: (data: any) => void,
): () => void {
  if (!Notifications) return () => {};

  const subscription = Notifications.addNotificationResponseReceivedListener(
    (response: any) => {
      handler(response.notification.request.content.data || {});
    },
  );

  return () => subscription?.remove();
}

// ── Badge ───────────────────────────────────────────────────────────────────

export async function setBadgeCount(count: number): Promise<void> {
  if (Notifications) {
    await Notifications.setBadgeCountAsync(count);
  }
}

export async function getBadgeCount(): Promise<number> {
  if (Notifications) {
    return Notifications.getBadgeCountAsync();
  }
  return 0;
}

// ── Topics ──────────────────────────────────────────────────────────────────

export async function subscribeToTopic(topic: string): Promise<void> {
  const token = await getStoredPushToken();
  if (token) {
    try {
      // Expo doesn't have native topic subscription — handled server-side
      // Just update the device registration
      await api.post('/mobile/notifications/register', {
        token,
        platform: Platform.OS,
        topics: ['chat', 'training', topic],
      });
    } catch (e) {
      if (__DEV__) console.warn('[push-notifications] topic subscribe failed:', e);
    }
  }
}
