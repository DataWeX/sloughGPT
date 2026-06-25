import React, {useState} from 'react';
import {View, Text, ScrollView, TouchableOpacity, StyleSheet} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {colors, spacing, radii, typography} from '../theme';

const FAQ = [
  {
    q: 'How do I start chatting?',
    a: 'Go to the Models tab, load a model (GPT-2 or Qwen recommended), then switch to Chat. Pick a personality from the soul pills and start typing.',
  },
  {
    q: 'How do I train my own model?',
    a: 'Go to the Train tab. Paste training text or select a dataset. Set epochs and learning rate, then tap Start Training. Watch the loss curve in real-time.',
  },
  {
    q: 'What models work best?',
    a: 'For chat: Qwen2.5-0.5B-Instruct (500M, stable on CPU). For training: GPT-2 (124M, fast). Smaller models run faster on mobile.',
  },
  {
    q: 'What is a "soul"?',
    a: 'A soul is a personality preset that changes how the AI responds. Switch between assistant, creative, coder, teacher, or analyst in the Models tab.',
  },
  {
    q: 'How do I add knowledge?',
    a: 'Go to the Knowledge tab. Tap + Add to add items one by one, or Import to paste multiple items at once. Knowledge is sent to the AI as context.',
  },
  {
    q: 'What training text works?',
    a: 'Plain text, SRT subtitles, or any line-based format. The more text, the better. At least 50 characters recommended. Shakespeare, transcripts, or domain-specific text all work.',
  },
  {
    q: 'How do I export data?',
    a: 'Knowledge: tap Export to share as JSON. Chat: tap Export in the header to share as markdown. Training checkpoints are saved automatically.',
  },
  {
    q: 'Can I use this offline?',
    a: 'Chat requires a server connection. Training requires a server. Knowledge can be viewed offline if cached. On-device inference is planned via llama.rn.',
  },
];

function FAQItem({item}: {item: {q: string; a: string}}) {
  const [open, setOpen] = useState(false);
  return (
    <TouchableOpacity style={styles.faqItem} onPress={() => setOpen(!open)}>
      <View style={styles.faqHeader}>
        <Text style={styles.faqQ}>{item.q}</Text>
        <Text style={styles.faqArrow}>{open ? '−' : '+'}</Text>
      </View>
      {open && <Text style={styles.faqA}>{item.a}</Text>}
    </TouchableOpacity>
  );
}

export function HelpScreen() {
  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Help</Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Quick Start</Text>
          <Text style={styles.step}>1. Connect to your server (Settings → Server)</Text>
          <Text style={styles.step}>2. Load a model (Models tab → tap Load)</Text>
          <Text style={styles.step}>3. Pick a personality (Models tab → tap a soul)</Text>
          <Text style={styles.step}>4. Start chatting (Chat tab)</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Keyboard Shortcuts</Text>
          <View style={styles.shortcutRow}>
            <Text style={styles.shortcutKey}>Enter</Text>
            <Text style={styles.shortcutDesc}>Send message</Text>
          </View>
          <View style={styles.shortcutRow}>
            <Text style={styles.shortcutKey}>Long press</Text>
            <Text style={styles.shortcutDesc}>Message actions (copy, feedback, regenerate)</Text>
          </View>
          <View style={styles.shortcutRow}>
            <Text style={styles.shortcutKey}>Pull down</Text>
            <Text style={styles.shortcutDesc}>Refresh data</Text>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>FAQ</Text>
          {FAQ.map((item, i) => (
            <FAQItem key={i} item={item} />
          ))}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Troubleshooting</Text>
          <Text style={styles.troubleshoot}>
            • Connection refused — Make sure the server is running on the correct port.
          </Text>
          <Text style={styles.troubleshoot}>
            • Model won't load — Try a smaller model (GPT-2 or Qwen2.5-0.5B).
          </Text>
          <Text style={styles.troubleshoot}>
            • Training fails — Ensure you have at least 50 characters of training text.
          </Text>
          <Text style={styles.troubleshoot}>
            • App crashes — Restart the app. Check the error boundary for details.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {flex: 1, backgroundColor: colors.background},
  content: {padding: spacing.lg, gap: spacing.md},
  title: {...typography.h1, color: colors.text},
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    padding: spacing.lg,
  },
  cardTitle: {
    ...typography.h3,
    color: colors.text,
    marginBottom: spacing.md,
  },
  step: {
    ...typography.body,
    color: colors.text,
    paddingVertical: spacing.xs,
  },
  shortcutRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingVertical: spacing.xs,
  },
  shortcutKey: {
    ...typography.caption,
    color: colors.primary,
    fontWeight: '600',
    minWidth: 80,
  },
  shortcutDesc: {
    ...typography.body,
    color: colors.textSecondary,
    flex: 1,
  },
  faqItem: {
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    paddingVertical: spacing.md,
  },
  faqHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  faqQ: {
    ...typography.body,
    color: colors.text,
    fontWeight: '500',
    flex: 1,
  },
  faqArrow: {
    fontSize: 20,
    color: colors.primary,
    fontWeight: '700',
  },
  faqA: {
    ...typography.caption,
    color: colors.textSecondary,
    marginTop: spacing.sm,
    lineHeight: 20,
  },
  troubleshoot: {
    ...typography.caption,
    color: colors.textSecondary,
    paddingVertical: spacing.xs,
    lineHeight: 20,
  },
});
