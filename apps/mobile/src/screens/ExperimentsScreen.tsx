import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl, TextInput as RNTextInput} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';

interface Experiment {
  id: string;
  name?: string;
  runs?: number;
  status?: string;
}

interface ExperimentData {
  id: string;
  metrics: Array<{metric: string; value: number; step: number; timestamp: string}>;
  params: Array<{param: string; value: string; timestamp: string}>;
  status: {status: string; completed_at?: string} | null;
}

export function ExperimentsScreen() {
  const colors = useColors();
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [newName, setNewName] = useState('');
  const [creating, setCreating] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedData, setSelectedData] = useState<ExperimentData | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const data = await api.get<{experiments: string[]}>('/experiments').catch(() => ({experiments: []}));
      setExperiments((data.experiments ?? []).map(id => ({id})));
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

  const handleCreate = async () => {
    if (!newName.trim()) return;
    try {
      setCreating(true);
      triggerHaptic('light');
      await api.post('/experiments', {name: newName.trim()});
      triggerHaptic('success');
      toast.success('Experiment created');
      setNewName('');
      await fetchData();
    } catch {
      toast.error('Failed to create experiment');
    } finally {
      setCreating(false);
    }
  };

  const handleSelect = async (id: string) => {
    setSelectedId(id);
    try {
      const data = await api.get<ExperimentData>(`/experiments/${encodeURIComponent(id)}/data`);
      setSelectedData(data);
    } catch {
      setSelectedData(null);
      toast.error('Failed to load experiment data');
    }
  };

  const handleDelete = async (id: string) => {
    try {
      triggerHaptic('light');
      await api.delete(`/experiments/${encodeURIComponent(id)}`);
      triggerHaptic('success');
      toast.success('Experiment deleted');
      if (selectedId === id) {
        setSelectedId(null);
        setSelectedData(null);
      }
      await fetchData();
    } catch {
      toast.error('Failed to delete experiment');
    }
  };

  const accent = colors.primary;

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <YStack>
          <Text fontSize={20} fontWeight="600" color={colors.text}>Experiments</Text>
          <Text fontSize={12} color={colors.textSecondary}>
            {experiments.length} experiments
          </Text>
        </YStack>
        <Pressable onPress={onRefresh} style={{padding: 8}}>
          <Icon name="refresh-cw" size={20} color={accent} />
        </Pressable>
      </XStack>

      {/* Create Form */}
      <YStack paddingHorizontal={16} marginBottom={12}>
        <YStack backgroundColor={colors.backgroundHover} borderRadius={8} padding={12}>
          <Text fontSize={13} fontWeight="500" color={colors.text} marginBottom={8}>New Experiment</Text>
          <XStack gap={8}>
            <RNTextInput
              value={newName}
              onChangeText={setNewName}
              placeholder="Experiment name..."
              placeholderTextColor={colors.textMuted}
              style={{
                flex: 1,
                backgroundColor: colors.background,
                borderRadius: 8,
                paddingHorizontal: 12,
                paddingVertical: 8,
                fontSize: 14,
                color: colors.text,
              }}
            />
            <Pressable
              onPress={handleCreate}
              disabled={creating || !newName.trim()}
              style={{
                backgroundColor: creating || !newName.trim() ? colors.textMuted : accent,
                borderRadius: 8,
                paddingHorizontal: 16,
                paddingVertical: 8,
                alignItems: 'center',
                justifyContent: 'center',
              }}>
              <Icon name="plus" size={18} color="#fff" />
            </Pressable>
          </XStack>
        </YStack>
      </YStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <Icon name="refresh-cw" size={24} color={colors.textSecondary} />
          <Text fontSize={13} color={colors.textSecondary} marginTop={8}>Loading experiments...</Text>
        </YStack>
      ) : (
        <FlatList
          data={experiments}
          keyExtractor={item => item.id}
          contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 20}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={accent} />}
          ListEmptyComponent={
            <YStack alignItems="center" paddingVertical={40}>
              <Icon name="layers" size={32} color={colors.textSecondary} />
              <Text fontSize={14} color={colors.textSecondary} marginTop={8}>No experiments yet</Text>
              <Text fontSize={12} color={colors.textMuted} marginTop={4}>Create one above to start tracking</Text>
            </YStack>
          }
          renderItem={({item}) => {
            const isSelected = selectedId === item.id;
            return (
              <YStack marginBottom={8}>
                <Pressable
                  onPress={() => handleSelect(item.id)}
                  style={{width: '100%'}}>
                  <YStack
                    backgroundColor={isSelected ? `${accent}15` : colors.backgroundHover}
                    borderRadius={8}
                    padding={12}
                    borderWidth={isSelected ? 1 : 0}
                    borderColor={accent}>
                    <XStack justifyContent="space-between" alignItems="center">
                      <YStack flex={1}>
                        <Text fontSize={13} fontWeight="500" color={colors.text}>
                          {item.name || item.id}
                        </Text>
                        {item.name && item.name !== item.id && (
                          <Text fontSize={11} color={colors.textMuted} marginTop={2}>{item.id}</Text>
                        )}
                      </YStack>
                      <XStack alignItems="center" gap={8}>
                        {item.status && (
                          <StatusBadge
                            label={item.status}
                            variant={item.status === 'completed' ? 'success' : item.status === 'active' ? 'info' : 'default'}
                          />
                        )}
                        <Pressable
                          onPress={() => handleDelete(item.id)}
                          style={{padding: 4}}>
                          <Icon name="trash-2" size={16} color={colors.error} />
                        </Pressable>
                      </XStack>
                    </XStack>
                  </YStack>
                </Pressable>

                {/* Expanded Details */}
                {isSelected && selectedData && (
                  <YStack
                    backgroundColor={colors.backgroundHover}
                    borderRadius={8}
                    padding={12}
                    marginTop={4}
                    borderLeftWidth={3}
                    borderLeftColor={accent}>
                    {/* Status */}
                    {selectedData.status && (
                      <XStack marginBottom={8} alignItems="center" gap={8}>
                        <Text fontSize={12} color={colors.textSecondary}>Status:</Text>
                        <StatusBadge
                          label={selectedData.status.status}
                          variant={selectedData.status.status === 'completed' ? 'success' : 'info'}
                        />
                        {selectedData.status.completed_at && (
                          <Text fontSize={11} color={colors.textMuted}>
                            Completed {new Date(selectedData.status.completed_at).toLocaleDateString()}
                          </Text>
                        )}
                      </XStack>
                    )}

                    {/* Metrics */}
                    {selectedData.metrics.length > 0 && (
                      <YStack marginBottom={8}>
                        <Text fontSize={12} fontWeight="500" color={colors.textSecondary} marginBottom={4}>Metrics</Text>
                        {selectedData.metrics.map((m, i) => (
                          <XStack key={i} justifyContent="space-between" paddingVertical={2}>
                            <Text fontSize={12} color={colors.text}>{m.metric}</Text>
                            <Text fontSize={12} fontWeight="500" color={accent}>
                              {typeof m.value === 'number' ? m.value.toFixed(4) : m.value}
                            </Text>
                          </XStack>
                        ))}
                      </YStack>
                    )}

                    {/* Params */}
                    {selectedData.params.length > 0 && (
                      <YStack>
                        <Text fontSize={12} fontWeight="500" color={colors.textSecondary} marginBottom={4}>Parameters</Text>
                        {selectedData.params.map((p, i) => (
                          <XStack key={i} justifyContent="space-between" paddingVertical={2}>
                            <Text fontSize={12} color={colors.text}>{p.param}</Text>
                            <Text fontSize={12} color={colors.textMuted}>{p.value}</Text>
                          </XStack>
                        ))}
                      </YStack>
                    )}
                  </YStack>
                )}
              </YStack>
            );
          }}
        />
      )}
    </SafeAreaView>
  );
}
