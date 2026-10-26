import React, { useState } from 'react'
import { Modal, TextInput, ActivityIndicator, Pressable } from 'react-native'
import { YStack, XStack, Text } from 'tamagui'
import { useColors } from '../../theme/colors'
import { Icon } from '../Icon'
import { useHapticPress } from '../../hooks/useHapticPress'
import { triggerHaptic } from '../../services/haptics'
import { toast } from '../../services/toast'
import { searchDatasets } from '../../services/training-service'
import type {
  DatasetSearchProviderStatus,
  DatasetSearchResult,
} from '../../services/training-service'
import { useTrainingStore } from '../../stores/training-store'

type ImportSource = 'url' | 'github' | 'huggingface' | 'csv'

const IMPORT_SOURCES: { key: ImportSource; label: string; placeholder: string }[] = [
  { key: 'url', label: 'URL', placeholder: 'https://example.com/data.txt' },
  { key: 'github', label: 'GitHub', placeholder: 'owner/repo or full URL' },
  { key: 'huggingface', label: 'HuggingFace', placeholder: 'dataset-id or org/dataset' },
  { key: 'csv', label: 'CSV', placeholder: 'https://example.com/data.csv' },
]

interface Props {
  visible: boolean
  onClose: () => void
}

export function ImportDatasetModal({ visible, onClose }: Props) {
  const colors = useColors()
  const hapticPress = useHapticPress()
  const importDataset = useTrainingStore((s) => s.importDataset)
  const [importSource, setImportSource] = useState('')
  const [importName, setImportName] = useState('')
  const [importType, setImportType] = useState<ImportSource>('url')
  const [importing, setImporting] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<DatasetSearchResult[]>([])
  const [searchProviders, setSearchProviders] = useState<DatasetSearchProviderStatus[]>([])

  const handleImport = async () => {
    if (!importSource.trim()) {
      return
    }
    setImporting(true)
    try {
      await importDataset(importSource.trim(), importName.trim(), importType)
      triggerHaptic('success')
      toast.success('Dataset imported successfully')
      onClose()
      setImportSource('')
      setImportName('')
    } catch (err: any) {
      toast.error(err.message || 'Failed to import dataset')
    } finally {
      setImporting(false)
    }
  }

  const handleSearch = async () => {
    const q = searchQuery.trim()
    if (!q) {
      return
    }
    setSearching(true)
    try {
      const res = await searchDatasets(q)
      setSearchResults(res.results ?? [])
      setSearchProviders(res.providers ?? [])
    } catch (err: any) {
      setSearchResults([])
      setSearchProviders([])
      toast.error(err.message || 'Search failed')
    } finally {
      setSearching(false)
    }
  }

  const selectResult = (r: DatasetSearchResult) => {
    // Mobile import surface is url/github/huggingface/csv — kaggle and
    // openlibrary rows render but never select (web carries those fields).
    if (r.source === 'github') {
      setImportType('github')
      setImportSource(r.id)
    } else if (r.source === 'huggingface') {
      setImportType('huggingface')
      setImportSource(r.id)
    } else {
      return
    }
    const idPart = r.id.includes('/') ? r.id.split('/').pop()! : r.id
    setImportName(idPart)
    setSearchResults([])
  }

  return (
    <Modal visible={visible} animationType="slide" transparent>
      <YStack flex={1} backgroundColor={colors.overlay(0.4)} justifyContent="flex-end">
        <YStack
          backgroundColor={colors.background}
          borderTopLeftRadius={24}
          borderTopRightRadius={24}
        >
          <XStack
            alignItems="center"
            justifyContent="space-between"
            paddingHorizontal={20}
            paddingVertical={16}
            borderBottomWidth={1}
            borderBottomColor="$borderColor"
          >
            <Text fontSize={16} fontWeight="600" color={colors.text}>
              Import Dataset
            </Text>
            <Pressable onPress={hapticPress('light', onClose)} accessibilityLabel="Close import">
              <YStack
                width={28}
                height={28}
                borderRadius={9}
                alignItems="center"
                justifyContent="center"
              >
                <Icon name="x" size={16} color={colors.textSecondary} />
              </YStack>
            </Pressable>
          </XStack>
          <YStack padding={20} gap={16}>
            <YStack
              gap={8}
              padding={12}
              borderRadius={12}
              borderWidth={0.5}
              borderColor={colors.primaryAlpha(0.3)}
              backgroundColor={colors.primaryAlpha(0.04)}
              accessibilityLabel="Search all sources"
            >
              <Text fontSize={13} color={colors.textSecondary} fontWeight="500">
                Search all sources
              </Text>
              <XStack gap={8} alignItems="center">
                <TextInput
                  value={searchQuery}
                  onChangeText={setSearchQuery}
                  placeholder="dataset name or topic"
                  placeholderTextColor={colors.textMuted}
                  autoCapitalize="none"
                  autoCorrect={false}
                  onSubmitEditing={handleSearch}
                  testID="federated-search-input"
                  accessibilityLabel="Federated search query"
                  style={{
                    flex: 1,
                    fontSize: 15,
                    color: colors.text,
                    backgroundColor: colors.background,
                    borderRadius: 8,
                    paddingHorizontal: 12,
                    paddingVertical: 8,
                  }}
                />
                <YStack
                  paddingHorizontal={12}
                  paddingVertical={8}
                  borderRadius={8}
                  backgroundColor={
                    searching || !searchQuery.trim() ? colors.primaryAlpha(0.3) : colors.primary
                  }
                  onPress={hapticPress('light', handleSearch)}
                  disabled={searching || !searchQuery.trim()}
                  pressStyle={{ opacity: 0.7 }}
                  testID="federated-search-button"
                  accessibilityLabel="Search all sources button"
                >
                  {searching ? (
                    <ActivityIndicator color={colors.white} />
                  ) : (
                    <Text fontSize={13} fontWeight="600" color={colors.white}>
                      Search
                    </Text>
                  )}
                </YStack>
              </XStack>
              {searchProviders
                .filter((p) => !p.ok)
                .map((p) => (
                  <Text
                    key={p.source}
                    fontSize={11}
                    color={colors.textMuted}
                    accessibilityLabel={`provider-failed-${p.source}`}
                  >
                    {p.source}: {p.error ?? 'unavailable'}
                  </Text>
                ))}
              {searchResults.map((r) => {
                const selectable = r.source === 'github' || r.source === 'huggingface'
                return (
                  <Pressable
                    key={`${r.source}:${r.id}`}
                    onPress={
                      selectable ? hapticPress('selection', () => selectResult(r)) : undefined
                    }
                    disabled={!selectable}
                    testID={`federated-result-${r.source}`}
                    accessibilityLabel={`Select ${r.title} from ${r.source}`}
                  >
                    <YStack
                      gap={2}
                      paddingVertical={6}
                      paddingHorizontal={4}
                      opacity={selectable ? 1 : 0.5}
                    >
                      <XStack gap={6} alignItems="center">
                        <Text fontSize={12} fontWeight="600" color={colors.primary}>
                          {r.source}
                        </Text>
                        <Text
                          fontSize={13}
                          color={colors.text}
                          numberOfLines={1}
                          style={{ flex: 1 }}
                        >
                          {r.title}
                        </Text>
                      </XStack>
                      <Text fontSize={11} color={colors.textMuted} numberOfLines={1}>
                        {r.id}
                        {r.description ? ` — ${r.description}` : ''}
                      </Text>
                      {!selectable && (
                        <Text fontSize={10} color={colors.textMuted}>
                          not importable on mobile
                        </Text>
                      )}
                    </YStack>
                  </Pressable>
                )
              })}
            </YStack>
            <YStack gap={8}>
              <Text fontSize={13} color={colors.textSecondary} fontWeight="500">
                Source
              </Text>
              <XStack gap={4} flexWrap="wrap">
                {IMPORT_SOURCES.map((src) => (
                  <YStack
                    key={src.key}
                    paddingHorizontal={12}
                    paddingVertical={4}
                    borderRadius={999}
                    backgroundColor={importType === src.key ? colors.primary : '$background'}
                    borderWidth={0.5}
                    borderColor={importType === src.key ? colors.primary : '$borderColor'}
                    onPress={hapticPress('selection', () => {
                      setImportType(src.key)
                      setImportSource('')
                    })}
                    pressStyle={{ opacity: 0.7 }}
                  >
                    <Text
                      fontSize={11}
                      color={importType === src.key ? colors.white : colors.textMuted}
                      letterSpacing={0.2}
                    >
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
                testID="import-source-input"
                placeholder={IMPORT_SOURCES.find((s) => s.key === importType)?.placeholder}
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
                importing || !importSource.trim() ? colors.primaryAlpha(0.3) : colors.primary
              }
              onPress={hapticPress('light', handleImport)}
              disabled={importing || !importSource.trim()}
              pressStyle={{ opacity: 0.7 }}
            >
              {importing ? (
                <ActivityIndicator color={colors.white} />
              ) : (
                <Text fontSize={14} fontWeight="600" color={colors.white}>
                  Import
                </Text>
              )}
            </YStack>
          </YStack>
        </YStack>
      </YStack>
    </Modal>
  )
}
