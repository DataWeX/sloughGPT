import React, {useState} from 'react';
import {Modal, TextInput, ActivityIndicator, Pressable} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../../theme/colors';
import {Icon} from '../Icon';
import {useHapticPress} from '../../hooks/useHapticPress';
import {triggerHaptic} from '../../services/haptics';
import {toast} from '../../services/toast';
import {useTrainingStore} from '../../stores/training-store';

type ImportSource = 'url' | 'github' | 'huggingface' | 'csv';

const IMPORT_SOURCES: {key: ImportSource; label: string; placeholder: string}[] = [
  {key: 'url', label: 'URL', placeholder: 'https://example.com/data.txt'},
  {key: 'github', label: 'GitHub', placeholder: 'owner/repo or full URL'},
  {key: 'huggingface', label: 'HuggingFace', placeholder: 'dataset-id or org/dataset'},
  {key: 'csv', label: 'CSV', placeholder: 'https://example.com/data.csv'},
];

interface Props {
  visible: boolean;
  onClose: () => void;
}

export function ImportDatasetModal({visible, onClose}: Props) {
  const colors = useColors();
  const hapticPress = useHapticPress();
  const importDataset = useTrainingStore(s => s.importDataset);
  const [importSource, setImportSource] = useState('');
  const [importName, setImportName] = useState('');
  const [importType, setImportType] = useState<ImportSource>('url');
  const [importing, setImporting] = useState(false);

  const handleImport = async () => {
    if (!importSource.trim()) {
      return;
    }
    setImporting(true);
    try {
      await importDataset(importSource.trim(), importName.trim(), importType);
      triggerHaptic('success');
      toast.success('Dataset imported successfully');
      onClose();
      setImportSource('');
      setImportName('');
    } catch (err: any) {
      toast.error(err.message || 'Failed to import dataset');
    } finally {
      setImporting(false);
    }
  };

  return (
    <Modal visible={visible} animationType="slide" transparent>
      <YStack
        flex={1}
        backgroundColor={colors.overlay(0.4)}
        justifyContent="flex-end">
        <YStack
          backgroundColor={colors.background}
          borderTopLeftRadius={24}
          borderTopRightRadius={24}>
          <XStack
            alignItems="center"
            justifyContent="space-between"
            paddingHorizontal={20}
            paddingVertical={16}
            borderBottomWidth={1}
            borderBottomColor="$borderColor">
            <Text fontSize={16} fontWeight="600" color={colors.text}>
              Import Dataset
            </Text>
            <Pressable
              onPress={hapticPress('light', onClose)}
              accessibilityLabel="Close import">
              <YStack
                width={28}
                height={28}
                borderRadius={9}
                alignItems="center"
                justifyContent="center">
                <Icon name="x" size={16} color={colors.textSecondary} />
              </YStack>
            </Pressable>
          </XStack>
          <YStack padding={20} gap={16}>
            <YStack gap={8}>
              <Text
                fontSize={13}
                color={colors.textSecondary}
                fontWeight="500">
                Source
              </Text>
              <XStack gap={4} flexWrap="wrap">
                {IMPORT_SOURCES.map(src => (
                  <YStack
                    key={src.key}
                    paddingHorizontal={12}
                    paddingVertical={4}
                    borderRadius={999}
                    backgroundColor={
                      importType === src.key ? colors.primary : '$background'
                    }
                    borderWidth={0.5}
                    borderColor={
                      importType === src.key ? colors.primary : '$borderColor'
                    }
                    onPress={hapticPress('selection', () => {
                      setImportType(src.key);
                      setImportSource('');
                    })}
                    pressStyle={{opacity: 0.7}}>
                    <Text
                      fontSize={11}
                      color={
                        importType === src.key ? colors.white : colors.textMuted
                      }
                      letterSpacing={0.2}>
                      {src.label}
                    </Text>
                  </YStack>
                ))}
              </XStack>
            </YStack>
            <YStack gap={4}>
              <Text fontSize={13} color={colors.textSecondary}>
                {importType === 'github'
                  ? 'Repository'
                  : importType === 'huggingface'
                  ? 'Dataset ID'
                  : 'URL or Path'}
              </Text>
              <TextInput
                value={importSource}
                onChangeText={setImportSource}
                placeholder={
                  IMPORT_SOURCES.find(s => s.key === importType)?.placeholder
                }
                placeholderTextColor={colors.textMuted}
                autoCapitalize="none"
                autoCorrect={false}
                style={{
                  fontSize: 15,
                  color: colors.text,
                  backgroundColor: colors.primaryAlpha(0.04),
                  borderRadius: 8,
                  paddingHorizontal: 12,
                  paddingVertical: 10,
                }}
              />
            </YStack>
            <YStack gap={4}>
              <Text fontSize={13} color={colors.textSecondary}>
                Name (optional)
              </Text>
              <TextInput
                value={importName}
                onChangeText={setImportName}
                placeholder="my-dataset"
                placeholderTextColor={colors.textMuted}
                autoCapitalize="none"
                autoCorrect={false}
                style={{
                  fontSize: 15,
                  color: colors.text,
                  backgroundColor: colors.primaryAlpha(0.04),
                  borderRadius: 8,
                  paddingHorizontal: 12,
                  paddingVertical: 10,
                }}
              />
            </YStack>
            <YStack
              paddingVertical={12}
              borderRadius={10}
              alignItems="center"
              backgroundColor={
                importing || !importSource.trim()
                  ? colors.primaryAlpha(0.3)
                  : colors.primary
              }
              onPress={hapticPress('light', handleImport)}
              disabled={importing || !importSource.trim()}
              pressStyle={{opacity: 0.7}}>
              {importing ? (
                <ActivityIndicator color={colors.white} />
              ) : (
                <Text
                  fontSize={14}
                  fontWeight="600"
                  color={colors.white}>
                  Import
                </Text>
              )}
            </YStack>
          </YStack>
        </YStack>
      </YStack>
    </Modal>
  );
}
