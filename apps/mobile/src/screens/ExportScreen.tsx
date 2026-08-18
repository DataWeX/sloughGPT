import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl, Alert} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';

interface ExportFormat {
  id: string;
  name: string;
  description: string;
}

interface Checkpoint {
  name: string;
  soul: string;
  loss: number;
  steps: number;
  created_at: string;
}

export function ExportScreen() {
  const colors = useColors();
  const [formats, setFormats] = useState<ExportFormat[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedFormat, setSelectedFormat] = useState<string | null>(null);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [f, c] = await Promise.all([
        api.get<{formats: ExportFormat[]}>('/models/export/formats').catch(() => ({formats: []})),
        api.get<{checkpoints: Checkpoint[]}>('/auto-train/checkpoints').catch(() => ({checkpoints: []})),
      ]);
      setFormats(f.formats || []);
      setCheckpoints(c.checkpoints || []);
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

  const handleExportModel = async () => {
    if (!selectedFormat) {
      toast.info('Select a format first');
      return;
    }
    try {
      setExporting(true);
      const result = await api.post<{path: string}>('/models/export', {format: selectedFormat});
      triggerHaptic('success');
      toast.success(`Exported to ${result.path}`);
    } catch {
      toast.error('Export failed');
    } finally {
      setExporting(false);
    }
  };

  const handleDownloadCheckpoint = async (name: string) => {
    try {
      const blob = await api.get<Blob>(`/auto-train/checkpoints/${encodeURIComponent(name)}/download`);
      triggerHaptic('success');
      toast.success(`Downloaded ${name}`);
    } catch {
      toast.error('Download failed');
    }
  };

  const formatLoss = (loss: number) => loss?.toFixed(3) ?? '—';

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Export</Text>
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
            <YStack padding={16} gap={16}>
              {/* Model Export */}
              <YStack gap={8}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>Model Export</Text>
                {formats.length === 0 ? (
                  <Text fontSize={13} color={colors.textMuted}>No export formats available</Text>
                ) : (
                  <YStack gap={6}>
                    {formats.map(f => (
                      <Pressable key={f.id} onPress={() => setSelectedFormat(f.id)}>
                        <XStack
                          padding={10}
                          borderRadius={8}
                          borderWidth={1}
                          borderColor={selectedFormat === f.id ? colors.primary : colors.border}
                          backgroundColor={selectedFormat === f.id ? colors.primary + '10' : colors.white}
                          gap={8}
                          alignItems="center">
                          <Icon name="download" size={16} color={selectedFormat === f.id ? colors.primary : colors.textMuted} />
                          <YStack flex={1}>
                            <Text fontSize={13} fontWeight="500" color={colors.text}>{f.name}</Text>
                            <Text fontSize={11} color={colors.textMuted}>{f.description}</Text>
                          </YStack>
                        </XStack>
                      </Pressable>
                    ))}
                  </YStack>
                )}
                <Pressable onPress={handleExportModel} disabled={!selectedFormat || exporting}>
                  <XStack
                    padding={10}
                    borderRadius={8}
                    backgroundColor={selectedFormat && !exporting ? colors.primary : colors.border}
                    alignItems="center"
                    justifyContent="center"
                    gap={6}
                    opacity={selectedFormat && !exporting ? 1 : 0.5}>
                    <Icon name="download" size={16} color="white" />
                    <Text fontSize={13} fontWeight="600" color="white">
                      {exporting ? 'Exporting...' : 'Export Model'}
                    </Text>
                  </XStack>
                </Pressable>
              </YStack>

              {/* Checkpoints */}
              <YStack gap={8}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>Checkpoints</Text>
                {checkpoints.length === 0 ? (
                  <Text fontSize={13} color={colors.textMuted}>No checkpoints available</Text>
                ) : (
                  checkpoints.map(cp => (
                    <XStack
                      key={cp.name}
                      padding={10}
                      borderRadius={8}
                      borderWidth={0.5}
                      borderColor={colors.border}
                      backgroundColor={colors.white}
                      gap={8}
                      alignItems="center">
                      <YStack flex={1} gap={2}>
                        <Text fontSize={13} fontWeight="500" color={colors.text}>{cp.name}</Text>
                        <XStack gap={6}>
                          <StatusBadge label={cp.soul} variant="info" />
                          <StatusBadge label={`loss: ${formatLoss(cp.loss)}`} variant="default" />
                        </XStack>
                      </YStack>
                      <Pressable onPress={() => handleDownloadCheckpoint(cp.name)}>
                        <XStack paddingHorizontal={8} paddingVertical={4} borderRadius={6} backgroundColor={colors.primary + '15'} gap={4} alignItems="center">
                          <Icon name="download" size={14} color={colors.primary} />
                          <Text fontSize={11} fontWeight="500" color={colors.primary}>Download</Text>
                        </XStack>
                      </Pressable>
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
