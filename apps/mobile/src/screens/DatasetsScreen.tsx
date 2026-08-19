import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl, Alert, TextInput, Modal} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useNavigation} from '@react-navigation/native';
import type {NativeStackNavigationProp} from '@react-navigation/native-stack';
import type {ToolsStackParamList} from '../navigation/types';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';
import type {Dataset} from '../types';

export function DatasetsScreen() {
  const colors = useColors();
  const navigation = useNavigation<NativeStackNavigationProp<ToolsStackParamList>>();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [importing, setImporting] = useState(false);
  const [urlModalVisible, setUrlModalVisible] = useState(false);
  const [urlInput, setUrlInput] = useState('');

  const fetchDatasets = useCallback(async () => {
    try {
      const data = await api.get<{datasets: Dataset[]}>('/datasets');
      setDatasets(data.datasets || []);
    } catch {
      setDatasets([]);
    }
  }, []);

  useEffect(() => {
    fetchDatasets().finally(() => setLoading(false));
  }, [fetchDatasets]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchDatasets();
    setRefreshing(false);
  };

  const handleImportLocal = async () => {
    Alert.alert('Import Dataset', 'On mobile, import a dataset by URL. Server-local file paths are not accessible from mobile.', [
      {text: 'Cancel', style: 'cancel'},
      {text: 'Import from URL', onPress: () => setUrlModalVisible(true)},
    ]);
  };

  const handleImportURL = async () => {
    const url = urlInput.trim();
    if (!url) return;
    try {
      setImporting(true);
      setUrlModalVisible(false);
      setUrlInput('');
      await api.post('/datasets/import/url', {url, name: url.split('/').pop() || 'url-dataset'});
      triggerHaptic('success');
      toast.success('Dataset imported');
      await fetchDatasets();
    } catch {
      toast.error('Import failed');
    } finally {
      setImporting(false);
    }
  };

  const handleDelete = (id: string, name: string) => {
    Alert.alert('Delete Dataset', `Delete "${name}"?`, [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            await api.delete(`/datasets/${id}`);
            triggerHaptic('success');
            toast.success('Deleted');
            await fetchDatasets();
          } catch {
            toast.error('Delete failed');
          }
        },
      },
    ]);
  };

  const formatSize = (bytes: number) => {
    if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${bytes} B`;
  };

  const renderItem = ({item}: {item: Dataset}) => (
    <Pressable onPress={() => navigation.navigate('DatasetDetail', {datasetId: item.id})}>
    <XStack
      padding={12}
      borderRadius={10}
      borderWidth={0.5}
      borderColor={colors.border}
      backgroundColor={colors.white}
      gap={10}
      alignItems="center">
      <YStack width={36} height={36} borderRadius={8} backgroundColor={colors.primary + '15'} alignItems="center" justifyContent="center">
        <Icon name="package" size={18} color={colors.primary} />
      </YStack>
      <YStack flex={1} gap={2}>
        <Text fontSize={14} fontWeight="500" color={colors.text}>{item.name}</Text>
        <XStack gap={8}>
          <StatusBadge label={item.format || 'jsonl'} variant="default" />
          {item.row_count > 0 && <StatusBadge label={`${item.row_count} rows`} variant="info" />}
          {item.total_chars > 0 && <StatusBadge label={formatSize(item.total_chars)} variant="default" />}
        </XStack>
        {item.description ? (
          <Text fontSize={12} color={colors.textMuted} numberOfLines={1}>{item.description}</Text>
        ) : null}
      </YStack>
      <Pressable onPress={() => handleDelete(item.id, item.name)}>
        <Icon name="trash-2" size={16} color={colors.error} />
      </Pressable>
    </XStack>
    </Pressable>
  );

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Datasets</Text>
        <XStack gap={8}>
          <Pressable onPress={() => setUrlModalVisible(true)}>
            <XStack paddingHorizontal={10} paddingVertical={5} borderRadius={6} backgroundColor={colors.primary + '15'} gap={4} alignItems="center">
              <Icon name="external-link" size={14} color={colors.primary} />
              <Text fontSize={12} fontWeight="500" color={colors.primary}>URL</Text>
            </XStack>
          </Pressable>
          <Pressable onPress={handleImportLocal} disabled={importing}>
            <XStack paddingHorizontal={10} paddingVertical={5} borderRadius={6} backgroundColor={colors.primary} gap={4} alignItems="center" opacity={importing ? 0.5 : 1}>
              <Icon name="plus" size={14} color="white" />
              <Text fontSize={12} fontWeight="500" color="white">{importing ? 'Importing...' : 'Import'}</Text>
            </XStack>
          </Pressable>
        </XStack>
      </XStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      ) : datasets.length === 0 ? (
        <YStack flex={1} alignItems="center" justifyContent="center" gap={8}>
          <Icon name="package" size={32} color={colors.textMuted} />
          <Text fontSize={14} color={colors.textMuted}>No datasets</Text>
          <Text fontSize={12} color={colors.textMuted}>Import a dataset to get started</Text>
        </YStack>
      ) : (
        <FlatList
          data={datasets}
          keyExtractor={item => item.id}
          renderItem={renderItem}
          contentContainerStyle={{padding: 16, gap: 8}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        />
      )}

      <Modal visible={urlModalVisible} transparent animationType="fade">
        <YStack flex={1} justifyContent="center" alignItems="center" backgroundColor={colors.overlay(0.4)}>
          <YStack
            width="85%"
            backgroundColor={colors.white}
            borderRadius={16}
            padding={20}
            gap={16}>
            <Text fontSize={17} fontWeight="600" color={colors.text}>Import from URL</Text>
            <TextInput
              value={urlInput}
              onChangeText={setUrlInput}
              placeholder="https://example.com/dataset.jsonl"
              placeholderTextColor={colors.textMuted}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              style={{
                borderWidth: 1,
                borderColor: colors.border,
                borderRadius: 10,
                padding: 12,
                fontSize: 15,
                color: colors.text,
                backgroundColor: colors.background,
              }}
            />
            <XStack gap={10} justifyContent="flex-end">
              <Pressable
                onPress={() => {
                  setUrlModalVisible(false);
                  setUrlInput('');
                }}>
                <XStack paddingHorizontal={16} paddingVertical={8} borderRadius={8}>
                  <Text fontSize={14} color={colors.textMuted}>Cancel</Text>
                </XStack>
              </Pressable>
              <Pressable
                onPress={handleImportURL}
                disabled={!urlInput.trim()}>
                <XStack
                  paddingHorizontal={16}
                  paddingVertical={8}
                  borderRadius={8}
                  backgroundColor={urlInput.trim() ? colors.primary : colors.border}
                  opacity={urlInput.trim() ? 1 : 0.5}>
                  <Text fontSize={14} fontWeight="500" color="white">Import</Text>
                </XStack>
              </Pressable>
            </XStack>
          </YStack>
        </YStack>
      </Modal>
    </SafeAreaView>
  );
}
