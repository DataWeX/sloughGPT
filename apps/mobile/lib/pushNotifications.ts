import { useState, useEffect, useRef } from 'react';
import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';
import { Analytics, PerformanceTracker } from './analytics';

// Configure how notifications should be handled when the app is in the foreground
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
    shouldShowBanner: true,
    shouldShowList: true,
  }),
});

export interface UsePushNotificationsReturn {
  expoPushToken: string | null;
  notification: Notifications.Notification | null;
  registerForPushNotifications: () => Promise<void>;
}

export function usePushNotifications(): UsePushNotificationsReturn {
  const [expoPushToken, setExpoPushToken] = useState<string | null>(null);
  const [notification, setNotification] = useState<Notifications.Notification | null>(null);
  const notificationListener = useRef<Notifications.Subscription | null>(null);
  const responseListener = useRef<Notifications.Subscription | null>(null);

  useEffect(() => {
    // Listen for notifications when app is in foreground
    notificationListener.current = Notifications.addNotificationReceivedListener(notification => {
      setNotification(notification);
      Analytics.trackEvent('notification_received', {
        title: notification.request.content.title,
      });
    });

    // Listen for notification responses (user tapped notification)
    responseListener.current = Notifications.addNotificationResponseReceivedListener(response => {
      Analytics.trackEvent('notification_tapped', {
        title: response.notification.request.content.title,
      });
      // Handle navigation or other actions here
    });

    return () => {
      notificationListener.current?.remove()
      responseListener.current?.remove()
    };
  }, []);

  const registerForPushNotifications = async () => {
    try {
      if (Platform.OS === 'web') {
        Analytics.trackEvent('push_notifications_unsupported_web');
        return;
      }

      const { status: existingStatus } = await Notifications.getPermissionsAsync();
      let finalStatus = existingStatus;

      if (existingStatus !== 'granted') {
        const { status } = await Notifications.requestPermissionsAsync();
        finalStatus = status;
      }

      if (finalStatus !== 'granted') {
        Analytics.trackEvent('push_permission_denied');
        return;
      }

      Analytics.trackEvent('push_permission_granted');

      // Get push token
      const token = (await Notifications.getExpoPushTokenAsync()).data;
      setExpoPushToken(token);
      
      Analytics.trackEvent('push_token_received', { token: token.substring(0, 20) + '...' });

      // Send token to your backend here
      // await apiClient.post('/mobile/push-token', { token });

      if (Platform.OS === 'android') {
        await Notifications.setNotificationChannelAsync('default', {
          name: 'default',
          importance: Notifications.AndroidImportance.MAX,
          vibrationPattern: [0, 250, 250, 250],
          lightColor: '#7C52C4',
        });
      }
    } catch (error) {
      PerformanceTracker.trackError(error as Error, { context: 'push_registration' });
    }
  };

  return {
    expoPushToken,
    notification,
    registerForPushNotifications,
  };
}

// Helper function to schedule local notifications
export async function scheduleLocalNotification(
  title: string,
  body: string,
  data?: Record<string, any>,
  trigger?: Notifications.NotificationTriggerInput
) {
  try {
    await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data,
        sound: true,
      },
      trigger: trigger || null,
    });
    Analytics.trackEvent('local_notification_scheduled', { title });
  } catch (error) {
    PerformanceTracker.trackError(error as Error, { context: 'schedule_notification' });
  }
}
