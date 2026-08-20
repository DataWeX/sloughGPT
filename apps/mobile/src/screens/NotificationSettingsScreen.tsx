import React, {useState, useEffect, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl, Alert} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text, Switch} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';
import {
  registerForPushNotifications,
  unregisterPushNotifications,
  isNotificationsEnabled,
  subscribeToTopic,
  unsubscribeFromTopic,
  getSubscribedTopics,
  onNotification,
} from '../services/push-notifications';

interface HistoryEntry {
  timestamp: number;
  title: string;
  body: string;
  topic: string | null;
  sent: number;
}

const TOPICS = [
  {id: 'chat', label: 'Chat replies', desc: 'When your assistant responds'},
  {id: 'training', label: 'Training updates', desc: 'Training start, progress, and completion'},
];

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffH = Math.floor(diffMin / 60);
  if (diffH < 24) return `${diffH}h ago`;
  return d.toLocaleDateString();
}

export function NotificationSettingsScreen() {
  const colors = useColors();
  const [enabled, setEnabled] = useState(false);
  const [subscribedTopics, setSubscribedTopics] = useState<string[]>(['chat', 'training']);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [quietHours, setQuietHours] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [isEnabled, topics, hist] = await Promise.all([
        isNotificationsEnabled(),
        getSubscribedTopics(),
        api.get<{history: HistoryEntry[]}>('/mobile/notifications/history?limit=20').catch(() => ({history: []})),
      ]);
      setEnabled(isEnabled);
      setSubscribedTopics(topics);
      setHistory(hist.history || []);
    } catch {
      // handled above
    }
  }, []);

  useEffect(() => {
    fetchData().finally(() => setLoading(false));
  }, [fetchData]);

  useEffect(() => {
    return onNotification((_title, _body) => {
      fetchData();
    });
  }, [fetchData]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
  };

  const handleToggleMaster = async (val: boolean) => {
    try {
      if (val) {
        const token = await registerForPushNotifications();
        setEnabled(!!token);
        if (token) {
          triggerHaptic('success');
          toast.success('Notifications enabled');
        } else {
          Alert.alert('Permission denied', 'Enable notifications in system settings.');
        }
      } else {
        await unregisterPushNotifications();
        setEnabled(false);
        triggerHaptic('light');
        toast.info('Notifications disabled');
      }
    } catch {
      toast.error('Failed to update notification settings');
    }
  };

  const handleToggleTopic = async (topicId: string, val: boolean) => {
    try {
      if (val) {
        await subscribeToTopic(topicId);
        setSubscribedTopics(prev => [...new Set([...prev, topicId])]);
      } else {
        await unsubscribeFromTopic(topicId);
        setSubscribedTopics(prev => prev.filter(t => t !== topicId));
      }
      triggerHaptic('light');
    } catch {
      toast.error('Failed to update topic');
    }
  };

  const handleTestNotification = async () => {
    try {
      await api.post('/mobile/notifications/send', {
        title: 'Test Notification',
        body: 'This is a test from SloughGPT',
        topic: 'chat',
        data: {screen: 'Chat'},
      });
      triggerHaptic('success');
      toast.success('Test notification sent');
    } catch {
      toast.error('Failed to send test notification');
    }
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Notifications</Text>
        <Pressable onPress={onRefresh}>
          <Icon name="refresh-cw" size={18} color={colors.primary} />
        </Pressable>
      </XStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      ) : (
        <FlatList
          data={[]}
          renderItem={() => null}
          ListHeaderComponent={
            <YStack padding={16} gap={20}>
              {/* Master Toggle */}
              <YStack gap={8}>
                <XStack
                  padding={14}
                  borderRadius={10}
                  borderWidth={0.5}
                  borderColor={colors.border}
                  backgroundColor={colors.white}
                  alignItems="center"
                  justifyContent="space-between">
                  <YStack gap={2}>
                    <Text fontSize={15} fontWeight="600" color={colors.text}>Push Notifications</Text>
                    <Text fontSize={12} color={colors.textMuted}>Receive alerts on this device</Text>
                  </YStack>
                  <Switch
                    size="small"
                    checked={enabled}
                    onCheckedChange={handleToggleMaster}
                    backgroundColor={enabled ? colors.primary : colors.border}
                  />
                </XStack>
              </YStack>

              {/* Per-Topic Toggles */}
              {enabled && (
                <YStack gap={8}>
                  <Text fontSize={13} fontWeight="600" color={colors.textMuted} textTransform="uppercase" letterSpacing={0.5}>
                    Topics
                  </Text>
                  {TOPICS.map(topic => {
                    const isOn = subscribedTopics.includes(topic.id);
                    return (
                      <XStack
                        key={topic.id}
                        padding={12}
                        borderRadius={10}
                        borderWidth={0.5}
                        borderColor={colors.border}
                        backgroundColor={colors.white}
                        alignItems="center"
                        justifyContent="space-between">
                        <YStack gap={1} flex={1}>
                          <Text fontSize={14} fontWeight="500" color={colors.text}>{topic.label}</Text>
                          <Text fontSize={11} color={colors.textMuted}>{topic.desc}</Text>
                        </YStack>
                        <Switch
                          size="small"
                          checked={isOn}
                          onCheckedChange={(val) => handleToggleTopic(topic.id, val)}
                          backgroundColor={isOn ? colors.primary : colors.border}
                        />
                      </XStack>
                    );
                  })}
                </YStack>
              )}

              {/* Quiet Hours */}
              {enabled && (
                <YStack gap={8}>
                  <Text fontSize={13} fontWeight="600" color={colors.textMuted} textTransform="uppercase" letterSpacing={0.5}>
                    Do Not Disturb
                  </Text>
                  <XStack
                    padding={12}
                    borderRadius={10}
                    borderWidth={0.5}
                    borderColor={colors.border}
                    backgroundColor={colors.white}
                    alignItems="center"
                    justifyContent="space-between">
                    <YStack gap={1} flex={1}>
                      <Text fontSize={14} fontWeight="500" color={colors.text}>Quiet Hours</Text>
                      <Text fontSize={11} color={colors.textMuted}>Mute notifications 10 PM – 8 AM</Text>
                    </YStack>
                    <Switch
                      size="small"
                      checked={quietHours}
                      onCheckedChange={setQuietHours}
                      backgroundColor={quietHours ? colors.primary : colors.border}
                    />
                  </XStack>
                </YStack>
              )}

              {/* Test Button */}
              {enabled && (
                <Pressable onPress={handleTestNotification}>
                  <XStack
                    padding={12}
                    borderRadius={10}
                    borderWidth={0.5}
                    borderColor={colors.primary + '40'}
                    backgroundColor={colors.primary + '08'}
                    alignItems="center"
                    justifyContent="center"
                    gap={8}>
                    <Icon name="upload" size={16} color={colors.primary} />
                    <Text fontSize={13} fontWeight="600" color={colors.primary}>Send Test Notification</Text>
                  </XStack>
                </Pressable>
              )}

              {/* History */}
              <YStack gap={8}>
                <Text fontSize={13} fontWeight="600" color={colors.textMuted} textTransform="uppercase" letterSpacing={0.5}>
                  Recent History
                </Text>
                {history.length === 0 ? (
                  <XStack padding={16} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} justifyContent="center">
                    <Text fontSize={13} color={colors.textMuted}>No notifications yet</Text>
                  </XStack>
                ) : (
                  history.map((entry, i) => (
                    <XStack
                      key={i}
                      padding={12}
                      borderRadius={10}
                      borderWidth={0.5}
                      borderColor={colors.border}
                      backgroundColor={colors.white}
                      gap={10}
                      alignItems="flex-start">
                      <YStack
                        width={32} height={32} borderRadius={8}
                        backgroundColor={colors.primaryAlpha(0.1)}
                        alignItems="center" justifyContent="center"
                        marginTop={2}>
                        <Icon name="zap" size={14} color={colors.primary} />
                      </YStack>
                      <YStack flex={1} gap={2}>
                        <XStack justifyContent="space-between" alignItems="center">
                          <Text fontSize={13} fontWeight="600" color={colors.text} numberOfLines={1}>{entry.title}</Text>
                          <Text fontSize={10} color={colors.textMuted}>{formatTime(entry.timestamp)}</Text>
                        </XStack>
                        <Text fontSize={12} color={colors.textMuted} numberOfLines={2}>{entry.body}</Text>
                        <XStack gap={6} marginTop={2}>
                          {entry.topic && <StatusBadge label={entry.topic} variant="info" />}
                          <StatusBadge label={`sent: ${entry.sent}`} variant="default" />
                        </XStack>
                      </YStack>
                    </XStack>
                  ))
                )}
              </YStack>
            </YStack>
          }
          contentContainerStyle={{paddingBottom: 32}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        />
      )}
    </SafeAreaView>
  );
}
