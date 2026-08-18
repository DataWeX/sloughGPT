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
  status: string;
  feedback_recorded: number;
  auto_train_steps: number;
  workflow_runs: number;
  last_run: string | null;
  aggregate_interval: number;
  prune_interval: number;
  export_interval: number;
  health_check_interval: number;
}

export function WorkflowScreen() {
  const colors = useColors();
  const [workflow, setWorkflow] = useState<WorkflowStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [toggling, setToggling] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const data = await api.get<WorkflowStatus>('/workflow/status');
      setWorkflow(data);
    } catch {
      setWorkflow(null);
    }
  }, []);

  useEffect(() => {
    fetchStatus().finally(() => setLoading(false));
  }, [fetchStatus]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchStatus();
    setRefreshing(false);
  };

  const handleToggle = async () => {
    const action = workflow?.status === 'running' ? 'stop' : 'start';
    try {
      setToggling(true);
      triggerHaptic('light');
      await api.post(`/workflow/${action}`);
      triggerHaptic('success');
      toast.success(`Workflow ${action}ed`);
      await fetchStatus();
    } catch {
      toast.error(`Failed to ${action} workflow`);
    } finally {
      setToggling(false);
    }
  };

  const handleTrigger = async (action: string) => {
    try {
      triggerHaptic('light');
      await api.post(`/workflow/trigger/${action}`);
      triggerHaptic('success');
      toast.success(`${action} triggered`);
    } catch {
      toast.error(`${action} failed`);
    }
  };

  const isRunning = workflow?.status === 'running';

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Workflow</Text>
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
              {/* Status + Toggle */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={10}>
                <XStack justifyContent="space-between" alignItems="center">
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Pipeline</Text>
                  <StatusBadge label={isRunning ? 'Running' : 'Stopped'} variant={isRunning ? 'success' : 'default'} />
                </XStack>
                <Pressable onPress={handleToggle} disabled={toggling}>
                  <XStack padding={10} borderRadius={8} backgroundColor={isRunning ? colors.error : colors.primary} alignItems="center" justifyContent="center" gap={6} opacity={toggling ? 0.5 : 1}>
                    <Icon name={isRunning ? 'stop-circle' : 'zap'} size={16} color="white" />
                    <Text fontSize={13} fontWeight="600" color="white">{toggling ? 'Toggling...' : isRunning ? 'Stop' : 'Start'}</Text>
                  </XStack>
                </Pressable>
              </YStack>

              {/* KPI Grid */}
              <XStack gap={8}>
                <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                  <Text fontSize={20} fontWeight="700" color={colors.text}>{workflow?.feedback_recorded ?? 0}</Text>
                  <Text fontSize={11} color={colors.textMuted}>Feedback</Text>
                </YStack>
                <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                  <Text fontSize={20} fontWeight="700" color={colors.text}>{workflow?.auto_train_steps ?? 0}</Text>
                  <Text fontSize={11} color={colors.textMuted}>Train Steps</Text>
                </YStack>
                <YStack flex={1} padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                  <Text fontSize={20} fontWeight="700" color={colors.text}>{workflow?.workflow_runs ?? 0}</Text>
                  <Text fontSize={11} color={colors.textMuted}>Runs</Text>
                </YStack>
              </XStack>

              {/* Configuration */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>Configuration</Text>
                {[
                  {label: 'Aggregate', value: workflow?.aggregate_interval},
                  {label: 'Prune', value: workflow?.prune_interval},
                  {label: 'Export', value: workflow?.export_interval},
                  {label: 'Health Check', value: workflow?.health_check_interval},
                ].map(item => (
                  <XStack key={item.label} justifyContent="space-between" alignItems="center">
                    <Text fontSize={13} color={colors.textMuted}>{item.label}</Text>
                    <Text fontSize={13} fontWeight="500" color={colors.text}>{item.value ?? '—'}s</Text>
                  </XStack>
                ))}
              </YStack>

              {/* Manual Triggers */}
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>Manual Triggers</Text>
                <XStack gap={8}>
                  {['aggregate', 'prune', 'export'].map(action => (
                    <Pressable key={action} onPress={() => handleTrigger(action)}>
                      <XStack paddingHorizontal={12} paddingVertical={6} borderRadius={6} backgroundColor={colors.primaryAlpha(0.1)} gap={4} alignItems="center">
                        <Icon name={action === 'aggregate' ? 'package' : action === 'prune' ? 'trash-2' : 'download'} size={14} color={colors.primary} />
                        <Text fontSize={12} fontWeight="500" color={colors.primary}>{action.charAt(0).toUpperCase() + action.slice(1)}</Text>
                      </XStack>
                    </Pressable>
                  ))}
                </XStack>
              </YStack>

              {/* Last Run */}
              {workflow?.last_run && (
                <YStack padding={12} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border}>
                  <XStack justifyContent="space-between" alignItems="center">
                    <Text fontSize={13} color={colors.textMuted}>Last Run</Text>
                    <Text fontSize={12} fontWeight="500" color={colors.text}>{new Date(workflow.last_run).toLocaleString()}</Text>
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
