import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';

interface FeedbackStats {
  thumbs_up: number;
  thumbs_down: number;
  total: number;
  up_ratio: number;
}

interface WorkflowStatus {
  active: boolean;
  last_run: string | null;
  pending_count: number;
  completed_count: number;
}

interface FeedbackItem {
  _id: string;
  message_id: string;
  rating: 'positive' | 'negative';
  user_message: string;
  assistant_response: string;
  timestamp: string;
  source?: string;
}

export function FeedbackScreen() {
  const colors = useColors();
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [workflow, setWorkflow] = useState<WorkflowStatus | null>(null);
  const [conversations, setConversations] = useState<FeedbackItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [triggering, setTriggering] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [s, w, c] = await Promise.all([
        api.get<FeedbackStats>('/feedback/stats/summary').catch(() => null),
        api.get<WorkflowStatus>('/workflow/status').catch(() => null),
        api.get<{conversations: FeedbackItem[]}>('/feedback/conversations?limit=50').catch(() => ({conversations: []})),
      ]);
      setStats(s);
      setWorkflow(w);
      setConversations(c.conversations ?? []);
    } catch {
      // handled above
    }
  }, []);

  useEffect(() => {
    fetchData().finally(() => setLoading(false));
  }, [fetchData]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
  };

  const handleTriggerWorkflow = async (action: string) => {
    try {
      setTriggering(true);
      triggerHaptic('light');
      await api.post(`/workflow/trigger/${action}`);
      triggerHaptic('success');
      toast.success(`Workflow ${action} triggered`);
      await fetchData();
    } catch {
      toast.error('Trigger failed');
    } finally {
      setTriggering(false);
    }
  };

  const positiveRate = stats && stats.total > 0 ? ((stats.thumbs_up / stats.total) * 100).toFixed(0) : '—';

  const formatTime = (ts: string) => {
    const d = new Date(ts);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 60000) return 'just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return d.toLocaleDateString();
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Feedback</Text>
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
          data={conversations}
          keyExtractor={item => item._id}
          contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 20}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
          ListHeaderComponent={
            <YStack gap={12} marginBottom={16}>
              {/* Stats Cards */}
              <XStack gap={8}>
                <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                  <Text fontSize={22} fontWeight="700" color={colors.text}>{stats?.total ?? 0}</Text>
                  <Text fontSize={11} color={colors.textMuted}>Total</Text>
                </YStack>
                <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                  <Text fontSize={22} fontWeight="700" color={colors.success}>{stats?.thumbs_up ?? 0}</Text>
                  <Text fontSize={11} color={colors.textMuted}>Positive</Text>
                </YStack>
                <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                  <Text fontSize={22} fontWeight="700" color={colors.error}>{stats?.thumbs_down ?? 0}</Text>
                  <Text fontSize={11} color={colors.textMuted}>Negative</Text>
                </YStack>
              </XStack>

              {/* Positive Rate */}
              <YStack padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4}>
                <XStack justifyContent="space-between" alignItems="center">
                  <Text fontSize={13} fontWeight="500" color={colors.text}>Approval Rate</Text>
                  <Text fontSize={18} fontWeight="700" color={colors.primary}>{positiveRate}%</Text>
                </XStack>
              </YStack>

              {/* Workflow Status */}
              <YStack padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
                <XStack justifyContent="space-between" alignItems="center">
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Workflow</Text>
                  <StatusBadge
                    label={workflow?.active ? 'Active' : 'Inactive'}
                    variant={workflow?.active ? 'success' : 'default'}
                  />
                </XStack>
                {workflow && (
                  <XStack gap={12}>
                    <YStack gap={2}>
                      <Text fontSize={11} color={colors.textMuted}>Pending</Text>
                      <Text fontSize={14} fontWeight="500" color={colors.text}>{workflow.pending_count}</Text>
                    </YStack>
                    <YStack gap={2}>
                      <Text fontSize={11} color={colors.textMuted}>Completed</Text>
                      <Text fontSize={14} fontWeight="500" color={colors.text}>{workflow.completed_count}</Text>
                    </YStack>
                    {workflow.last_run && (
                      <YStack gap={2}>
                        <Text fontSize={11} color={colors.textMuted}>Last Run</Text>
                        <Text fontSize={12} color={colors.text}>{new Date(workflow.last_run).toLocaleDateString()}</Text>
                      </YStack>
                    )}
                  </XStack>
                )}
                <XStack gap={8}>
                  <Pressable onPress={() => handleTriggerWorkflow('aggregate')} disabled={triggering}>
                    <XStack paddingHorizontal={10} paddingVertical={5} borderRadius={6} backgroundColor={colors.primary + '15'} gap={4} alignItems="center">
                      <Icon name="package" size={14} color={colors.primary} />
                      <Text fontSize={12} fontWeight="500" color={colors.primary}>Aggregate</Text>
                    </XStack>
                  </Pressable>
                  <Pressable onPress={() => handleTriggerWorkflow('evaluate')} disabled={triggering}>
                    <XStack paddingHorizontal={10} paddingVertical={5} borderRadius={6} backgroundColor={colors.primary + '15'} gap={4} alignItems="center">
                      <Icon name="zap" size={14} color={colors.primary} />
                      <Text fontSize={12} fontWeight="500" color={colors.primary}>Evaluate</Text>
                    </XStack>
                  </Pressable>
                </XStack>
              </YStack>

              {/* Conversations Header */}
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize={15} fontWeight="600" color={colors.text}>Recent Feedback</Text>
                <Text fontSize={12} color={colors.textMuted}>{conversations.length} items</Text>
              </XStack>
            </YStack>
          }
          ListEmptyComponent={
            <YStack alignItems="center" paddingVertical={40}>
              <Icon name="check" size={32} color={colors.success} />
              <Text fontSize={14} color={colors.textSecondary} marginTop={8}>No feedback yet</Text>
            </YStack>
          }
          renderItem={({item}) => (
            <YStack
              backgroundColor={colors.white}
              borderRadius={8}
              padding={12}
              marginBottom={8}
              borderWidth={0.5}
              borderColor={colors.border}>
              <XStack justifyContent="space-between" alignItems="center" marginBottom={6}>
                <StatusBadge
                  label={item.rating === 'positive' ? 'Positive' : 'Negative'}
                  variant={item.rating === 'positive' ? 'success' : 'error'}
                />
                <Text fontSize={11} color={colors.textMuted}>{formatTime(item.timestamp)}</Text>
              </XStack>
              {item.user_message ? (
                <Text fontSize={12} color={colors.textMuted} numberOfLines={2} marginBottom={4}>
                  User: {item.user_message}
                </Text>
              ) : null}
              {item.assistant_response ? (
                <Text fontSize={12} color={colors.text} numberOfLines={2}>
                  Assistant: {item.assistant_response}
                </Text>
              ) : null}
            </YStack>
          )}
        />
      )}
    </SafeAreaView>
  );
}
