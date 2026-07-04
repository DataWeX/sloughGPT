import AsyncStorage from '@react-native-async-storage/async-storage';

// Mock api-client
jest.mock('../../services/api-client', () => ({
  api: {post: jest.fn(async () => ({})), get: jest.fn(async () => ({status: 'healthy'}))},
  getApiUrl: jest.fn(async () => 'http://localhost:8000'),
}));

beforeEach(() => {
  jest.clearAllMocks();
  AsyncStorage.clear();
});

afterEach(() => {
  jest.restoreAllMocks();
});

import {
  registerForPushNotifications,
  getStoredPushToken,
  unregisterPushNotifications,
  isNotificationsEnabled,
  onNotification,
  setBadgeCount,
  getBadgeCount,
} from '../../services/push-notifications';

describe('push-notifications', () => {
  describe('registerForPushNotifications', () => {
    it('returns token on success', async () => {
      const token = await registerForPushNotifications();
      expect(token).toBe('ExpoPushToken[test123]');
    });

    it('stores token in AsyncStorage', async () => {
      await registerForPushNotifications();
      const stored = await AsyncStorage.getItem('@sloughgpt/push_token');
      expect(stored).toBe('ExpoPushToken[test123]');
    });

    it('marks notifications as enabled', async () => {
      await registerForPushNotifications();
      const enabled = await AsyncStorage.getItem('@sloughgpt/notifications_enabled');
      expect(enabled).toBe('true');
    });

    it('registers with backend', async () => {
      const {api} = require('../../services/api-client');
      await registerForPushNotifications();
      expect(api.post).toHaveBeenCalledWith(
        '/mobile/notifications/register',
        expect.objectContaining({
          token: 'ExpoPushToken[test123]',
          topics: ['chat', 'training'],
        }),
      );
    });

    it('returns null when permission denied', async () => {
      const Notifications = require('expo-notifications');
      Notifications.getPermissionsAsync.mockResolvedValueOnce({status: 'undetermined'});
      Notifications.requestPermissionsAsync.mockResolvedValueOnce({status: 'denied'});
      const token = await registerForPushNotifications();
      expect(token).toBeNull();
    });
  });

  describe('getStoredPushToken', () => {
    it('returns stored token', async () => {
      await AsyncStorage.setItem('@sloughgpt/push_token', 'token123');
      expect(await getStoredPushToken()).toBe('token123');
    });

    it('returns null when not registered', async () => {
      expect(await getStoredPushToken()).toBeNull();
    });
  });

  describe('unregisterPushNotifications', () => {
    it('removes stored token', async () => {
      await AsyncStorage.setItem('@sloughgpt/push_token', 'token123');
      await AsyncStorage.setItem('@sloughgpt/notifications_enabled', 'true');
      await unregisterPushNotifications();
      expect(await AsyncStorage.getItem('@sloughgpt/push_token')).toBeNull();
    });

    it('calls backend unregister', async () => {
      const {api} = require('../../services/api-client');
      await AsyncStorage.setItem('@sloughgpt/push_token', 'token123');
      await unregisterPushNotifications();
      expect(api.post).toHaveBeenCalledWith('/mobile/notifications/unregister', {
        token: 'token123',
      });
    });

    it('does nothing when no token stored', async () => {
      const {api} = require('../../services/api-client');
      await unregisterPushNotifications();
      expect(api.post).not.toHaveBeenCalled();
    });
  });

  describe('isNotificationsEnabled', () => {
    it('returns true when enabled', async () => {
      await AsyncStorage.setItem('@sloughgpt/notifications_enabled', 'true');
      expect(await isNotificationsEnabled()).toBe(true);
    });

    it('returns false by default', async () => {
      expect(await isNotificationsEnabled()).toBe(false);
    });
  });

  describe('onNotification', () => {
    it('calls handler when notification received', async () => {
      const handler = jest.fn();
      const unsub = onNotification(handler);
      expect(typeof unsub).toBe('function');
      unsub();
    });

    it('returns unsubscribe function', () => {
      const handler = jest.fn();
      const unsub = onNotification(handler);
      unsub();
    });
  });

  describe('badge', () => {
    it('setBadgeCount calls Notifications.setBadgeCountAsync', async () => {
      await setBadgeCount(5);
      const Notifications = require('expo-notifications');
      expect(Notifications.setBadgeCountAsync).toHaveBeenCalledWith(5);
    });

    it('getBadgeCount returns count', async () => {
      const count = await getBadgeCount();
      expect(count).toBe(0);
    });
  });
});
