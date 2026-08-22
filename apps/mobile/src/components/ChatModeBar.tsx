import React, {useState} from 'react';
import {Pressable, ScrollView, Modal, View} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {Icon} from './Icon';
import {triggerHaptic} from '../services/haptics';

export type ChatMode = 'chat' | 'write' | 'rewrite' | 'decide' | 'explain' | 'translate' | 'brainstorm' | 'wellness';

interface ModeOption {
  value: ChatMode;
  label: string;
  icon: string;
}

const MODES: ModeOption[] = [
  {value: 'chat', label: 'Chat', icon: 'message-circle'},
  {value: 'write', label: 'Write', icon: 'edit'},
  {value: 'rewrite', label: 'Rewrite', icon: 'refresh-cw'},
  {value: 'decide', label: 'Decide', icon: 'target'},
  {value: 'explain', label: 'Explain', icon: 'book-open'},
  {value: 'translate', label: 'Translate', icon: 'globe'},
  {value: 'brainstorm', label: 'Ideas', icon: 'zap'},
  {value: 'wellness', label: 'Wellness', icon: 'heart'},
];

const TONES = ['Friendly', 'Professional', 'Funny', 'Short', 'Detailed'];
const TYPES = ['Email', 'Social Post', 'Story', 'Poem', 'Letter', 'Note'];
const REWRITE_OPTIONS = ['Fix Grammar', 'Make Shorter', 'Make Friendlier', 'Make Professional', 'Sound Like Me'];
const DECIDE_STRUCTURES = ['Pros & Cons', 'Comparison', 'Simple Verdict', 'Deep Analysis'];
const DIFFICULTIES = ['Simple', 'Moderate', 'Expert'];
const LANG_PAIRS = ['EN→ES', 'EN→FR', 'EN→DE', 'EN→ZH', 'EN→JA', 'ES→EN', 'FR→EN'];
const BRAINSTORM_TOPICS = ['Name Ideas', 'Weekend Plans', 'Gift Ideas', 'Solve a Problem', 'Plan an Event'];
const WELLNESS_TYPES = ['Sleep Story', 'Meditation', 'Breathing', 'Affirmation'];

interface Props {
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
  tone: string;
  onToneChange: (tone: string) => void;
  type: string;
  onTypeChange: (type: string) => void;
  rewriteStyle: string;
  onRewriteStyleChange: (style: string) => void;
  decideStructure: string;
  onDecideStructureChange: (structure: string) => void;
  difficulty: string;
  onDifficultyChange: (difficulty: string) => void;
  langPair: string;
  onLangPairChange: (pair: string) => void;
  brainstormTopic: string;
  onBrainstormTopicChange: (topic: string) => void;
  wellnessType: string;
  onWellnessTypeChange: (type: string) => void;
}

function SubOptionPill({active, label, onPress}: {active: boolean; label: string; onPress: () => void}) {
  const colors = useColors();
  return (
    <Pressable
      onPress={onPress}
      hitSlop={8}
      style={({pressed}) => ({
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 12,
        backgroundColor: active
          ? colors.primaryAlpha(0.15)
          : pressed
            ? colors.primaryAlpha(0.05)
            : 'transparent',
        marginRight: 6,
      })}>
      <Text
        fontSize={11}
        fontWeight={active ? '600' : '400'}
        color={active ? colors.primary : colors.textMuted}>
        {label}
      </Text>
    </Pressable>
  );
}

export function ChatModeBar({
  mode, onModeChange,
  tone, onToneChange,
  type, onTypeChange,
  rewriteStyle, onRewriteStyleChange,
  decideStructure, onDecideStructureChange,
  difficulty, onDifficultyChange,
  langPair, onLangPairChange,
  brainstormTopic, onBrainstormTopicChange,
  wellnessType, onWellnessTypeChange,
}: Props) {
  const colors = useColors();
  const [showModes, setShowModes] = useState(false);
  const currentMode = MODES.find(m => m.value === mode) || MODES[0];

  const renderSubOptions = () => {
    switch (mode) {
      case 'write':
        return (
          <XStack gap={2} paddingVertical={6}>
            <Text fontSize={10} color={colors.textMuted} marginRight={4}>Tone</Text>
            {TONES.map(t => (
              <SubOptionPill key={t} active={tone === t} label={t} onPress={() => { onToneChange(t); triggerHaptic('light'); }} />
            ))}
            <View style={{width: 1, height: 14, backgroundColor: colors.border, marginHorizontal: 6, alignSelf: 'center'}} />
            <Text fontSize={10} color={colors.textMuted} marginRight={4}>Type</Text>
            {TYPES.map(t => (
              <SubOptionPill key={t} active={type === t} label={t} onPress={() => { onTypeChange(t); triggerHaptic('light'); }} />
            ))}
          </XStack>
        );
      case 'rewrite':
        return (
          <XStack gap={2} paddingVertical={6} flexWrap="wrap">
            {REWRITE_OPTIONS.map(o => (
              <SubOptionPill key={o} active={rewriteStyle === o} label={o} onPress={() => { onRewriteStyleChange(o); triggerHaptic('light'); }} />
            ))}
          </XStack>
        );
      case 'decide':
        return (
          <XStack gap={2} paddingVertical={6}>
            <Text fontSize={10} color={colors.textMuted} marginRight={4}>Output</Text>
            {DECIDE_STRUCTURES.map(s => (
              <SubOptionPill key={s} active={decideStructure === s} label={s} onPress={() => { onDecideStructureChange(s); triggerHaptic('light'); }} />
            ))}
          </XStack>
        );
      case 'explain':
        return (
          <XStack gap={2} paddingVertical={6}>
            <Text fontSize={10} color={colors.textMuted} marginRight={4}>Level</Text>
            {DIFFICULTIES.map(d => (
              <SubOptionPill key={d} active={difficulty === d} label={d} onPress={() => { onDifficultyChange(d); triggerHaptic('light'); }} />
            ))}
          </XStack>
        );
      case 'translate':
        return (
          <XStack gap={2} paddingVertical={6}>
            <Text fontSize={10} color={colors.textMuted} marginRight={4}>To</Text>
            {LANG_PAIRS.map(p => (
              <SubOptionPill key={p} active={langPair === p} label={p} onPress={() => { onLangPairChange(p); triggerHaptic('light'); }} />
            ))}
          </XStack>
        );
      case 'brainstorm':
        return (
          <XStack gap={2} paddingVertical={6}>
            <Text fontSize={10} color={colors.textMuted} marginRight={4}>Topic</Text>
            {BRAINSTORM_TOPICS.map(t => (
              <SubOptionPill key={t} active={brainstormTopic === t} label={t} onPress={() => { onBrainstormTopicChange(t); triggerHaptic('light'); }} />
            ))}
          </XStack>
        );
      case 'wellness':
        return (
          <XStack gap={2} paddingVertical={6}>
            <Text fontSize={10} color={colors.textMuted} marginRight={4}>Type</Text>
            {WELLNESS_TYPES.map(t => (
              <SubOptionPill key={t} active={wellnessType === t} label={t} onPress={() => { onWellnessTypeChange(t); triggerHaptic('light'); }} />
            ))}
          </XStack>
        );
      default:
        return null;
    }
  };

  return (
    <YStack borderBottomWidth={0} backgroundColor="$background">
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={{paddingHorizontal: 16, paddingVertical: 8, gap: 8}}>
        {MODES.map(m => (
          <Pressable
            key={m.value}
            onPress={() => { onModeChange(m.value); triggerHaptic('light'); }}
            hitSlop={8}
            style={({pressed}) => ({
              flexDirection: 'row',
              alignItems: 'center',
              gap: 6,
              paddingHorizontal: 14,
              paddingVertical: 8,
              borderRadius: 20,
              backgroundColor: mode === m.value
                ? colors.primary
                : pressed
                  ? colors.primaryAlpha(0.08)
                  : colors.primaryAlpha(0.04),
              borderWidth: mode === m.value ? 0 : 1,
              borderColor: colors.primaryAlpha(0.1),
            })}>
            <Icon
              name={m.icon as any}
              size={14}
              color={mode === m.value ? '#FFFFFF' : colors.textMuted}
            />
            <Text
              fontSize={13}
              fontWeight={mode === m.value ? '600' : '500'}
              color={mode === m.value ? '#FFFFFF' : colors.textMuted}>
              {m.label}
            </Text>
          </Pressable>
        ))}
      </ScrollView>
      {renderSubOptions()}
    </YStack>
  );
}
