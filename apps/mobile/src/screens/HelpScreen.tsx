import React, {useState} from 'react';
import {YStack, XStack, Text} from 'tamagui';
import {ScreenShell} from '../components/ScreenShell';
import {Icon} from '../components/Icon';
import {useColors} from '../theme/colors';

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
    <YStack
      borderBottomWidth={0.5}
      borderBottomColor="$borderColor"
      paddingVertical={12}
      onPress={() => setOpen(!open)}>
      <XStack alignItems="center" justifyContent="space-between">
        <Text flex={1} fontSize={14} color="$color" fontWeight="500">
          {item.q}
        </Text>
        <Text fontSize={18} color="$color9" fontWeight="700">
          {open ? '\u2212' : '+'}
        </Text>
      </XStack>
      {open && (
        <Text fontSize={13} color="$color11" marginTop={8} lineHeight={20}>
          {item.a}
        </Text>
      )}
    </YStack>
  );
}

export function HelpScreen() {
  const colors = useColors();
  return (
    <ScreenShell title="Help">
      <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
        <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
          Quick Start
        </Text>
        {['Connect to your server (Settings \u2192 Server)', 'Load a model (Models tab \u2192 tap Load)', 'Pick a personality (Models tab \u2192 tap a soul)', 'Start chatting (Chat tab)'].map((s, i) => (
          <Text key={i} fontSize={14} color="$color" paddingVertical={4}>
            {i + 1}. {s}
          </Text>
        ))}
      </YStack>

      <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
        <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
          Keyboard Shortcuts
        </Text>
        {[
          {key: 'Enter', desc: 'Send message'},
          {key: 'Long press', desc: 'Message actions (copy, good/bad, regenerate, delete)'},
          {key: 'Swipe left', desc: 'Delete message'},
          {key: 'Pull down', desc: 'Refresh data'},
        ].map((s, i) => (
          <XStack key={i} alignItems="center" gap={12} paddingVertical={4}>
            <Text minWidth={80} fontSize={13} fontWeight="600" color="$color9">
              {s.key}
            </Text>
            <Text flex={1} fontSize={14} color="$color11">
              {s.desc}
            </Text>
          </XStack>
        ))}
        <XStack alignItems="center" gap={12} paddingVertical={4}>
          <XStack minWidth={80} gap={4} alignItems="center">
            <Icon name="mic" size={14} color={colors.primary} />
            <Text fontSize={13} fontWeight="600" color="$color9">
              {' '}Button
            </Text>
          </XStack>
          <Text flex={1} fontSize={14} color="$color11">
            Voice input (tap to record, tap to stop & send)
          </Text>
        </XStack>
        <XStack alignItems="center" gap={12} paddingVertical={4}>
          <Text minWidth={80} fontSize={13} fontWeight="600" color="$color9">
            + Button
          </Text>
          <Text flex={1} fontSize={14} color="$color11">
            Attach image from gallery
          </Text>
        </XStack>
      </YStack>

      <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
        <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
          FAQ
        </Text>
        {FAQ.map((item, i) => (
          <FAQItem key={i} item={item} />
        ))}
      </YStack>

      <YStack backgroundColor="$background" borderRadius={12} padding={16} borderWidth={0.5} borderColor="$borderColor">
        <Text fontSize={16} fontWeight="600" color="$color" marginBottom={12}>
          Troubleshooting
        </Text>
        {[
          'Connection refused \u2014 Make sure the server is running on the correct port.',
          "Model won't load \u2014 Try a smaller model (GPT-2 or Qwen2.5-0.5B).",
          'Training fails \u2014 Ensure you have at least 50 characters of training text.',
          'App crashes \u2014 Restart the app. Check the error boundary for details.',
        ].map((s, i) => (
          <Text key={i} fontSize={13} color="$color11" paddingVertical={4} lineHeight={20}>
            {'\u2022'} {s}
          </Text>
        ))}
      </YStack>
    </ScreenShell>
  );
}
