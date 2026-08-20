import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, TextInput as RNTextInput, Image, RefreshControl, ActivityIndicator} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';

const API_URL_FALLBACK = 'http://localhost:8000';

interface GalleryImage {
  id: string;
  path: string;
  created: number;
}

interface Style {
  key: string;
  name: string;
}

const STYLES: Style[] = [
  {key: 'realistic', name: 'Realistic'},
  {key: 'cartoon', name: 'Cartoon'},
  {key: 'watercolor', name: 'Watercolor'},
  {key: 'sketch', name: 'Sketch'},
  {key: 'fantasy', name: 'Fantasy'},
];

export function ImagesScreen() {
  const colors = useColors();
  const [gallery, setGallery] = useState<GalleryImage[]>([]);
  const [styles, setStyles] = useState<Style[]>(STYLES);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [selectedStyle, setSelectedStyle] = useState('realistic');
  const [generating, setGenerating] = useState(false);
  const [lastGenerated, setLastGenerated] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [galleryRes, stylesRes] = await Promise.all([
        api.get<{images: GalleryImage[]}>('/images/gallery').catch(() => ({images: []})),
        api.get<{styles: [string, string][]}>('/images/styles').catch(() => ({styles: []})),
      ]);
      setGallery(galleryRes.images ?? []);
      if (stylesRes.styles?.length) {
        setStyles(stylesRes.styles.map(([k, n]) => ({key: k, name: n})));
      }
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

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    try {
      setGenerating(true);
      triggerHaptic('light');
      const result = await api.post<{image: string; style: string; prompt: string; id: string}>('/images/generate', {
        prompt: prompt.trim(),
        style: selectedStyle,
      });
      triggerHaptic('success');
      setLastGenerated(result.image);
      toast.success('Image generated');
      await fetchData();
    } catch {
      toast.error('Generation failed');
    } finally {
      setGenerating(false);
    }
  };

  const accent = colors.primary;

  const formatTime = (ts: number) => {
    const d = new Date(ts * 1000);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return d.toLocaleDateString();
  };

  const getImageUrl = (path: string) => {
    if (path.startsWith('http')) return path;
    return `${API_URL_FALLBACK}/${path}`;
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <YStack>
          <Text fontSize={20} fontWeight="600" color={colors.text}>Images</Text>
          <Text fontSize={12} color={colors.textSecondary}>
            {gallery.length} images generated
          </Text>
        </YStack>
        <Pressable onPress={onRefresh} style={{padding: 8}}>
          <Icon name="refresh-cw" size={20} color={accent} />
        </Pressable>
      </XStack>

      {/* Generation Form */}
      <YStack paddingHorizontal={16} marginBottom={12}>
        <YStack backgroundColor={colors.backgroundHover} borderRadius={8} padding={12}>
          <Text fontSize={13} fontWeight="500" color={colors.text} marginBottom={8}>Generate Image</Text>
          <RNTextInput
            value={prompt}
            onChangeText={setPrompt}
            placeholder="Describe the image you want to generate..."
            placeholderTextColor={colors.textMuted}
            multiline
            numberOfLines={3}
            style={{
              backgroundColor: colors.background,
              borderRadius: 8,
              padding: 12,
              fontSize: 14,
              color: colors.text,
              textAlignVertical: 'top',
              minHeight: 80,
              marginBottom: 8,
            }}
          />
          <XStack gap={4} marginBottom={8} flexWrap="wrap">
            {styles.map(s => (
              <Pressable
                key={s.key}
                onPress={() => {
                  setSelectedStyle(s.key);
                  triggerHaptic('light');
                }}
                style={{
                  paddingHorizontal: 10,
                  paddingVertical: 6,
                  borderRadius: 6,
                  backgroundColor: selectedStyle === s.key ? accent : colors.background,
                }}>
                <Text
                  fontSize={11}
                  fontWeight={selectedStyle === s.key ? '600' : '400'}
                  color={selectedStyle === s.key ? '#fff' : colors.text}>
                  {s.name}
                </Text>
              </Pressable>
            ))}
          </XStack>
          <Pressable
            onPress={handleGenerate}
            disabled={generating || !prompt.trim()}
            style={{
              backgroundColor: generating || !prompt.trim() ? colors.textMuted : accent,
              borderRadius: 8,
              paddingVertical: 10,
              alignItems: 'center',
              flexDirection: 'row',
              justifyContent: 'center',
              gap: 6,
            }}>
            {generating ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <Icon name="image" size={16} color="#fff" />
            )}
            <Text fontSize={13} fontWeight="600" color="#fff">
              {generating ? 'Generating...' : 'Generate'}
            </Text>
          </Pressable>
        </YStack>
      </YStack>

      {/* Last Generated */}
      {lastGenerated && (
        <YStack paddingHorizontal={16} marginBottom={12}>
          <YStack backgroundColor={colors.backgroundHover} borderRadius={8} padding={8}>
            <Text fontSize={12} fontWeight="500" color={colors.textSecondary} marginBottom={6}>Last Generated</Text>
            <Image
              source={{uri: getImageUrl(lastGenerated)}}
              style={{width: '100%', height: 200, borderRadius: 6}}
              resizeMode="cover"
            />
          </YStack>
        </YStack>
      )}

      {/* Gallery */}
      <XStack paddingHorizontal={16} marginBottom={8}>
        <Text fontSize={14} fontWeight="500" color={colors.text}>Gallery</Text>
      </XStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <Icon name="refresh-cw" size={24} color={colors.textSecondary} />
          <Text fontSize={13} color={colors.textSecondary} marginTop={8}>Loading gallery...</Text>
        </YStack>
      ) : (
        <FlatList
          data={gallery}
          keyExtractor={item => item.id}
          numColumns={2}
          columnWrapperStyle={{gap: 8, paddingHorizontal: 16}}
          contentContainerStyle={{paddingBottom: 20}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={accent} />}
          ListEmptyComponent={
            <YStack alignItems="center" paddingVertical={40}>
              <Icon name="image" size={32} color={colors.textSecondary} />
              <Text fontSize={14} color={colors.textSecondary} marginTop={8}>No images yet</Text>
              <Text fontSize={12} color={colors.textMuted} marginTop={4}>Generate your first image above</Text>
            </YStack>
          }
          renderItem={({item}) => (
            <YStack flex={1} marginBottom={8}>
              <Image
                source={{uri: getImageUrl(item.path)}}
                style={{width: '100%', height: 140, borderRadius: 6}}
                resizeMode="cover"
              />
              <Text fontSize={11} color={colors.textMuted} marginTop={4}>
                {formatTime(item.created)}
              </Text>
            </YStack>
          )}
        />
      )}
    </SafeAreaView>
  );
}
