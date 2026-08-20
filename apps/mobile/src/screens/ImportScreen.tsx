import React, {useState} from 'react';
import {FlatList, Pressable, Alert, Platform} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {getApiUrl} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';

let DocumentPicker: any = null;
try {
  DocumentPicker = require('expo-document-picker');
} catch {}

interface ImportItem {
  id: string;
  name: string;
  description: string;
  icon: string;
  accept: string;
  endpoint: string;
}

const IMPORT_ITEMS: ImportItem[] = [
  {
    id: 'settings',
    name: 'Settings',
    description: 'Import theme, language, and chat preferences',
    icon: 'settings',
    accept: '.json',
    endpoint: '/settings/import',
  },
  {
    id: 'checkpoint',
    name: 'Model Checkpoint',
    description: 'Import a trained model checkpoint',
    icon: 'box',
    accept: '.pt,.bin,.safetensors',
    endpoint: '/auto-train/checkpoints/import',
  },
  {
    id: 'dataset',
    name: 'Training Dataset',
    description: 'Import a JSONL training dataset',
    icon: 'database',
    accept: '.jsonl,.json',
    endpoint: '/datasets/import',
  },
  {
    id: 'soul',
    name: 'Soul Configuration',
    description: 'Import a soul personality definition',
    icon: 'user',
    accept: '.json',
    endpoint: '/souls/import',
  },
];

export function ImportScreen() {
  const colors = useColors();
  const [importing, setImporting] = useState<string | null>(null);
  const [lastImport, setLastImport] = useState<{id: string; time: number} | null>(null);

  const handleImport = async (item: ImportItem) => {
    if (!DocumentPicker) {
      toast.error('Document picker not available');
      return;
    }

    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: item.accept === '*' ? '*/*' : item.accept.split(',').map(e => {
          if (e === '.json') return 'application/json';
          if (e === '.jsonl') return 'application/x-ndjson';
          return 'application/octet-stream';
        }),
        copyToCacheDirectory: true,
      });

      if (result.canceled || !result.assets?.[0]) return;

      const file = result.assets[0];
      setImporting(item.id);
      triggerHaptic('light');

      const formData = new FormData();
      formData.append('file', {
        uri: file.uri,
        name: file.name,
        type: file.mimeType || 'application/octet-stream',
      } as any);

      const baseUrl = await getApiUrl();
      const res = await fetch(`${baseUrl}${item.endpoint}`, {
        method: 'POST',
        body: formData,
        headers: {'Content-Type': 'multipart/form-data'},
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }

      triggerHaptic('success');
      toast.success(`Imported ${item.name}`);
      setLastImport({id: item.id, time: Date.now()});
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Import failed';
      toast.error(msg);
      triggerHaptic('error');
    } finally {
      setImporting(null);
    }
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Import</Text>
        <Icon name="upload" size={18} color={colors.primary} />
      </XStack>

      <FlatList
        data={IMPORT_ITEMS}
        keyExtractor={item => item.id}
        contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 32, gap: 12}}
        renderItem={({item}) => {
          const isImporting = importing === item.id;
          const wasImported = lastImport?.id === item.id;

          return (
            <Pressable onPress={() => handleImport(item)} disabled={isImporting}>
              <XStack
                padding={14}
                borderRadius={10}
                borderWidth={1}
                borderColor={isImporting ? colors.primary : colors.border}
                backgroundColor={colors.white}
                gap={12}
                alignItems="center"
                opacity={isImporting ? 0.7 : 1}>
                <YStack
                  width={40}
                  height={40}
                  borderRadius={8}
                  backgroundColor={colors.primary + '12'}
                  alignItems="center"
                  justifyContent="center">
                  <Icon name={item.icon as any} size={18} color={colors.primary} />
                </YStack>
                <YStack flex={1} gap={2}>
                  <XStack alignItems="center" gap={6}>
                    <Text fontSize={14} fontWeight="500" color={colors.text}>{item.name}</Text>
                    {wasImported && (
                      <StatusBadge label="Done" variant="success" />
                    )}
                  </XStack>
                  <Text fontSize={12} color={colors.textMuted}>{item.description}</Text>
                </YStack>
                <Icon
                  name={isImporting ? 'refresh-cw' : 'upload'}
                  size={16}
                  color={isImporting ? colors.primary : colors.textMuted}
                />
              </XStack>
            </Pressable>
          );
        }}
      />
    </SafeAreaView>
  );
}
