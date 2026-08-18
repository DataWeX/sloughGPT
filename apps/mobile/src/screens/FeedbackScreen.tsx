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

interface WorkflowStatus {
  active: boolean;
  last_run: string | null;
  pending_count: number;
  completed_count: number;
}

interface TrainingStats {
  total_pairs: number;
  synced: number;
  pending: number;
}

export function FeedbackScreen() {
  const colors = useColors();
  const [stats, setStats] = useState<{total: number; positive: number; negative: number} | null>(null);
  const [workflow, setWorkflow] = useState<WorkflowStatus | null>(null);
  const [training, setTraining] = useState<TrainingStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [triggering, setTriggering] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [s, w, t] = await Promise.all([
        api.get<{total: number; positive: number; negative: number}>('/meta-weights/stats').catch(() => null),
        api.get<WorkflowStatus>('/workflow/status').catch(() => null),
        api.get<TrainingStats>('/training/status').catch(() => null),
      ]);
      setStats(s);
      setWorkflow(w);
      setTraining(t);
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

  const positiveRate = stats && stats.total > 0 ? ((stats.positive / stats.total) * 100).toFixed(0) : '—';

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
          data={[]}
          renderItem={() => null}
          ListHeaderComponent={
            <YStack padding={16} gap={12}>
              {/* Stats Cards */}
              <XStack gap={8}>
                <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                  <Text fontSize={22} fontWeight="700" color={colors.text}>{stats?.total ?? 0}</Text>
                  <Text fontSize={11} color={colors.textMuted}>Total</Text>
                </YStack>
                <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                  <Text fontSize={22} fontWeight="700" color={colors.success}>{stats?.positive ?? 0}</Text>
                  <Text fontSize={11} color={colors.textMuted}>Positive</Text>
                </YStack>
                <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                  <Text fontSize={22} fontWeight="700" color={colors.error}>{stats?.negative ?? 0}</Text>
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

              {/* Training Pairs */}
              {training && (
                <YStack padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Training Pairs</Text>
                  <XStack gap={12}>
                    <YStack gap={2}>
                      <Text fontSize={11} color={colors.textMuted}>Total</Text>
                      <Text fontSize={14} fontWeight="500" color={colors.text}>{training.total_pairs}</Text>
                    </YStack>
                    <YStack gap={2}>
                      <Text fontSize={11} color={colors.textMuted}>Synced</Text>
                      <Text fontSize={14} fontWeight="500" color={colors.success}>{training.synced}</Text>
                    </YStack>
                    <YStack gap={2}>
                      <Text fontSize={11} color={colors.textMuted}>Pending</Text>
                      <Text fontSize={14} fontWeight="500" color={colors.warning}>{training.pending}</Text>
                    </YStack>
                  </XStack>
                </YStack>
              )}
            </YStack>
          }
          contentContainerStyle={{paddingBottom: 32}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        />
      )}
    </SafeAreaView>
  );
}
