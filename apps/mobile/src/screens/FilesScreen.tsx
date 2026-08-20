import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, TextInput as RNTextInput, RefreshControl, Alert} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';

interface FileEntry {
  id: string;
  filename: string;
  size: number;
  content_type: string;
  uploaded_at: string;
  ingested: boolean;
  chunk_count?: number;
  tags?: string[];
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function getFileIcon(ext?: string) {
  switch (ext?.toLowerCase()) {
    case 'pdf': return 'book' as const;
    case 'txt': case 'md': return 'book' as const;
    case 'json': case 'csv': return 'book' as const;
    case 'py': case 'js': case 'ts': return 'terminal' as const;
    case 'png': case 'jpg': case 'jpeg': case 'gif': return 'image' as const;
    default: return 'book' as const;
  }
}

export function FilesScreen() {
  const colors = useColors();
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const data = await api.get<{files?: FileEntry[]} | FileEntry[]>('/files/').catch(() => []);
      const list = Array.isArray(data) ? data : (data?.files ?? []);
      setFiles(list);
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

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      await fetchData();
      return;
    }
    try {
      setSearching(true);
      const data = await api.get<{files?: FileEntry[]}>(`/files/search?q=${encodeURIComponent(searchQuery.trim())}`).catch(() => ({files: []}));
      setFiles(data.files ?? []);
    } catch {
      // handled above
    } finally {
      setSearching(false);
    }
  };

  const handleDelete = (file: FileEntry) => {
    Alert.alert('Delete File', `Delete "${file.filename}"?`, [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            triggerHaptic('light');
            await api.delete(`/files/${file.id}`);
            triggerHaptic('success');
            toast.success('File deleted');
            await fetchData();
          } catch {
            toast.error('Failed to delete file');
          }
        },
      },
    ]);
  };

  const handleIngest = async (file: FileEntry) => {
    try {
      triggerHaptic('light');
      await api.post(`/files/${file.id}/ingest`);
      triggerHaptic('success');
      toast.success('File ingested');
      await fetchData();
    } catch {
      toast.error('Failed to ingest file');
    }
  };

  const accent = colors.primary;

  const formatTime = (ts: string) => {
    const d = new Date(ts);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return d.toLocaleDateString();
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <YStack>
          <Text fontSize={20} fontWeight="600" color={colors.text}>Files</Text>
          <Text fontSize={12} color={colors.textSecondary}>
            {files.length} files
          </Text>
        </YStack>
        <Pressable onPress={onRefresh} style={{padding: 8}}>
          <Icon name="refresh-cw" size={20} color={accent} />
        </Pressable>
      </XStack>

      {/* Search */}
      <YStack paddingHorizontal={16} marginBottom={12}>
        <XStack gap={8}>
          <RNTextInput
            value={searchQuery}
            onChangeText={setSearchQuery}
            onSubmitEditing={handleSearch}
            placeholder="Search files..."
            placeholderTextColor={colors.textMuted}
            returnKeyType="search"
            style={{
              flex: 1,
              backgroundColor: colors.backgroundHover,
              borderRadius: 8,
              paddingHorizontal: 12,
              paddingVertical: 8,
              fontSize: 14,
              color: colors.text,
            }}
          />
          <Pressable
            onPress={handleSearch}
            disabled={searching}
            style={{
              backgroundColor: accent,
              borderRadius: 8,
              paddingHorizontal: 12,
              paddingVertical: 8,
              alignItems: 'center',
              justifyContent: 'center',
            }}>
            <Icon name="search" size={18} color="#fff" />
          </Pressable>
        </XStack>
      </YStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <Icon name="refresh-cw" size={24} color={colors.textSecondary} />
          <Text fontSize={13} color={colors.textSecondary} marginTop={8}>Loading files...</Text>
        </YStack>
      ) : (
        <FlatList
          data={files}
          keyExtractor={item => item.id}
          contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 20}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={accent} />}
          ListEmptyComponent={
            <YStack alignItems="center" paddingVertical={40}>
              <Icon name="book" size={32} color={colors.textSecondary} />
              <Text fontSize={14} color={colors.textSecondary} marginTop={8}>No files</Text>
              <Text fontSize={12} color={colors.textMuted} marginTop={4}>Upload files via the web interface</Text>
            </YStack>
          }
          renderItem={({item}) => (
            <YStack
              backgroundColor={colors.backgroundHover}
              borderRadius={8}
              padding={12}
              marginBottom={8}>
              <XStack justifyContent="space-between" alignItems="flex-start">
                <XStack gap={8} flex={1}>
                  <Icon name={getFileIcon(item.content_type)} size={18} color={accent} />
                  <YStack flex={1}>
                    <Text fontSize={13} fontWeight="500" color={colors.text} numberOfLines={1}>
                      {item.filename}
                    </Text>
                    <XStack gap={8} marginTop={2}>
                      <Text fontSize={11} color={colors.textMuted}>{formatSize(item.size)}</Text>
                      <Text fontSize={11} color={colors.textMuted}>{formatTime(item.uploaded_at)}</Text>
                    </XStack>
                  </YStack>
                </XStack>
                <XStack gap={4}>
                  {!item.ingested && (
                    <Pressable
                      onPress={() => handleIngest(item)}
                      style={{padding: 4}}>
                      <Icon name="download" size={16} color={colors.success} />
                    </Pressable>
                  )}
                  <Pressable
                    onPress={() => handleDelete(item)}
                    style={{padding: 4}}>
                    <Icon name="trash-2" size={16} color={colors.error} />
                  </Pressable>
                </XStack>
              </XStack>
              <XStack gap={4} marginTop={6}>
                <StatusBadge
                  label={item.ingested ? 'Ingested' : 'Pending'}
                  variant={item.ingested ? 'success' : 'warning'}
                />
                {item.chunk_count !== undefined && item.chunk_count > 0 && (
                  <StatusBadge label={`${item.chunk_count} chunks`} variant="info" />
                )}
              </XStack>
            </YStack>
          )}
        />
      )}
    </SafeAreaView>
  );
}
