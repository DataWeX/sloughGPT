import React from 'react';
import {render, waitFor} from '@/test-utils';

jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {
    SafeAreaView: ({children, style}: any) =>
      React.createElement(View, {style, testID: 'safe-area-view'}, children),
    useSafeAreaInsets: () => ({top: 0, bottom: 0, left: 0, right: 0}),
  };
});

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn().mockResolvedValue({history: []}),
    post: jest.fn().mockResolvedValue({status: 'sent'}),
  },
}));

jest.mock('../../services/push-notifications', () => ({
  registerForPushNotifications: jest.fn().mockResolvedValue('expo-token-123'),
  unregisterPushNotifications: jest.fn().mockResolvedValue(undefined),
  isNotificationsEnabled: jest.fn().mockResolvedValue(true),
  subscribeToTopic: jest.fn().mockResolvedValue(undefined),
  unsubscribeFromTopic: jest.fn().mockResolvedValue(undefined),
  getSubscribedTopics: jest.fn().mockResolvedValue(['chat', 'training']),
  onNotification: jest.fn().mockReturnValue(() => {}),
}));

jest.mock('../../services/haptics', () => ({triggerHaptic: jest.fn()}));

jest.mock('../../services/toast', () => ({
  toast: {success: jest.fn(), error: jest.fn(), info: jest.fn()},
}));

jest.mock('../../components/StatusBadge', () => ({
  StatusBadge: ({label}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, null, label);
  },
}));

jest.mock('../../components/Icon', () => ({
  Icon: ({name}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, null, name);
  },
}));

beforeEach(() => {
  jest.clearAllMocks();
});

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {NotificationSettingsScreen} = require('../NotificationSettingsScreen');

describe('NotificationSettingsScreen', () => {
  it('renders title', async () => {
    const {getByText} = await render(<NotificationSettingsScreen />);
    await waitFor(() => {
      expect(getByText('Notifications')).toBeTruthy();
    });
  });

  it('shows push notifications toggle', async () => {
    const {getByText} = await render(<NotificationSettingsScreen />);
    await waitFor(() => {
      expect(getByText('Push Notifications')).toBeTruthy();
    });
  });

  it('shows topic toggles when enabled', async () => {
    const {getByText} = await render(<NotificationSettingsScreen />);
    await waitFor(() => {
      expect(getByText('Chat replies')).toBeTruthy();
      expect(getByText('Training updates')).toBeTruthy();
    });
  });

  it('shows quiet hours toggle', async () => {
    const {getByText} = await render(<NotificationSettingsScreen />);
    await waitFor(() => {
      expect(getByText('Quiet Hours')).toBeTruthy();
    });
  });

  it('shows test notification button', async () => {
    const {getByText} = await render(<NotificationSettingsScreen />);
    await waitFor(() => {
      expect(getByText('Send Test Notification')).toBeTruthy();
    });
  });

  it('shows recent history section', async () => {
    const {getByText} = await render(<NotificationSettingsScreen />);
    await waitFor(() => {
      expect(getByText('Recent History')).toBeTruthy();
    });
  });

  it('shows empty state when no history', async () => {
    const {getByText} = await render(<NotificationSettingsScreen />);
    await waitFor(() => {
      expect(getByText('No notifications yet')).toBeTruthy();
    });
  });
});
