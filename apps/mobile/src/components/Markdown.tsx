import React, {useState, useCallback, useRef, useEffect} from 'react';
import {Clipboard, Pressable, View} from 'react-native';
import {Text, XStack, YStack, useTheme} from 'tamagui';
import {Icon} from './Icon';
import {toast} from '../services/toast';

interface MarkdownProps {
  content: string;
  streaming?: boolean;
}

type Block =
  | {type: 'heading'; level: number; text: string}
  | {type: 'code'; language: string; code: string}
  | {type: 'blockquote'; text: string}
  | {type: 'ul'; items: string[]}
  | {type: 'ol'; items: string[]}
  | {type: 'hr'}
  | {type: 'paragraph'; text: string};

function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  const lines = text.split('\n');
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    if (line.startsWith('```')) {
      const lang = line.slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      blocks.push({type: 'code', language: lang, code: codeLines.join('\n')});
      continue;
    }

    // Horizontal rule
    if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line.trim())) {
      blocks.push({type: 'hr'});
      i++;
      continue;
    }

    // Heading
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      blocks.push({type: 'heading', level: headingMatch[1].length, text: headingMatch[2]});
      i++;
      continue;
    }

    // Blockquote
    if (line.startsWith('>')) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].startsWith('>')) {
        quoteLines.push(lines[i].replace(/^>\s?/, ''));
        i++;
      }
      blocks.push({type: 'blockquote', text: quoteLines.join('\n')});
      continue;
    }

    // Unordered list
    if (/^[-*+]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*+]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*+]\s+/, ''));
        i++;
      }
      blocks.push({type: 'ul', items});
      continue;
    }

    // Ordered list
    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\.\s+/, ''));
        i++;
      }
      blocks.push({type: 'ol', items});
      continue;
    }

    // Empty line — skip
    if (line.trim() === '') {
      i++;
      continue;
    }

    // Paragraph — collect consecutive non-empty, non-special lines
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !lines[i].startsWith('```') &&
      !lines[i].startsWith('>') &&
      !/^#{1,6}\s/.test(lines[i]) &&
      !/^[-*+]\s+/.test(lines[i]) &&
      !/^\d+\.\s+/.test(lines[i]) &&
      !/^(-{3,}|\*{3,}|_{3,})\s*$/.test(lines[i].trim())
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    if (paraLines.length > 0) {
      blocks.push({type: 'paragraph', text: paraLines.join('\n')});
    }
  }

  return blocks;
}

function InlineText({text, fontSize = 14}: {text: string; fontSize?: number}) {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\)|~~[^~]+~~)/g).filter(Boolean);

  return (
    <Text fontSize={fontSize} lineHeight={fontSize * 1.5} color="$color">
      {parts.map((part, i) => {
        // Inline code
        if (part.startsWith('`') && part.endsWith('`') && part.length > 1) {
          return (
            <Text key={i} fontFamily="mono" fontSize={fontSize - 1}
              backgroundColor="rgba(124, 82, 196, 0.08)" paddingHorizontal={4}
              paddingVertical={1} borderRadius={3} color="$color9">
              {part.slice(1, -1)}
            </Text>
          );
        }
        // Bold
        if (part.startsWith('**') && part.endsWith('**')) {
          return <Text key={i} fontWeight="700">{part.slice(2, -2)}</Text>;
        }
        // Italic
        if (part.startsWith('*') && part.endsWith('*') && !part.startsWith('**')) {
          return <Text key={i} fontStyle="italic">{part.slice(1, -1)}</Text>;
        }
        // Strikethrough
        if (part.startsWith('~~') && part.endsWith('~~')) {
          return <Text key={i} textDecorationLine="line-through">{part.slice(2, -2)}</Text>;
        }
        // Link — groups are [text](url)
        if (part.startsWith('[')) {
          const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
          if (linkMatch) {
            return (
              <Text key={i} color="$color9" textDecorationLine="underline">
                {linkMatch[1]}
              </Text>
            );
          }
        }
        return <Text key={i}>{part}</Text>;
      })}
    </Text>
  );
}

function CodeBlock({language, code}: {language: string; code: string}) {
  const theme = useTheme();
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    Clipboard.setString(code);
    setCopied(true);
    toast.success('Copied to clipboard');
    setTimeout(() => setCopied(false), 1500);
  }, [code]);

  return (
    <YStack marginVertical={6} borderRadius={10} overflow="hidden"
      backgroundColor="rgba(26, 22, 37, 0.04)" borderWidth={0.5}
      borderColor="rgba(124, 82, 196, 0.1)">
      {/* Header bar */}
      <XStack paddingHorizontal={12} paddingVertical={6}
        backgroundColor="rgba(124, 82, 196, 0.06)"
        borderBottomWidth={0.5} borderBottomColor="rgba(124, 82, 196, 0.08)"
        alignItems="center" justifyContent="space-between">
        <Text fontSize={10} fontFamily="mono" fontWeight="500" color="$color10">
          {language || 'code'}
        </Text>
        <Pressable onPress={handleCopy} hitSlop={8}>
          <XStack alignItems="center" gap={4}>
            <Icon name={copied ? 'check' : 'copy'} size={12} color="$color10" />
            <Text fontSize={10} color="$color10">{copied ? 'Copied' : 'Copy'}</Text>
          </XStack>
        </Pressable>
      </XStack>
      {/* Code content */}
      <Text fontFamily="mono" fontSize={12} lineHeight={18}
        color="$color" paddingHorizontal={12} paddingVertical={10}
        selectable>
        {code}
      </Text>
    </YStack>
  );
}

export function Markdown({content, streaming = false}: MarkdownProps) {
  const [rendered, setRendered] = useState(content);
  const throttleRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Streaming throttle: don't re-parse on every token
  useEffect(() => {
    if (streaming) {
      if (throttleRef.current) return;
      throttleRef.current = setTimeout(() => {
        setRendered(content);
        throttleRef.current = null;
      }, 150);
    } else {
      if (throttleRef.current) {
        clearTimeout(throttleRef.current);
        throttleRef.current = null;
      }
      setRendered(content);
    }
    return () => {
      if (throttleRef.current) {
        clearTimeout(throttleRef.current);
        throttleRef.current = null;
      }
    };
  }, [content, streaming]);

  if (!rendered) return null;

  const blocks = parseBlocks(rendered);

  return (
    <YStack gap={2}>
      {blocks.map((block, i) => {
        switch (block.type) {
          case 'heading': {
            const sizes: Record<number, {size: number; weight: string; mt: number; mb: number}> = {
              1: {size: 20, weight: '700', mt: 12, mb: 4},
              2: {size: 18, weight: '600', mt: 10, mb: 4},
              3: {size: 16, weight: '600', mt: 8, mb: 2},
              4: {size: 15, weight: '600', mt: 6, mb: 2},
              5: {size: 14, weight: '600', mt: 4, mb: 2},
              6: {size: 13, weight: '600', mt: 4, mb: 2},
            };
            const s = sizes[block.level] || sizes[3];
            return (
              <Text key={i} fontSize={s.size} fontWeight={s.weight as any}
                letterSpacing={-0.3} color="$color" marginTop={s.mt} marginBottom={s.mb}>
                {block.text}
              </Text>
            );
          }
          case 'code':
            return <CodeBlock key={i} language={block.language} code={block.code} />;
          case 'blockquote':
            return (
              <XStack key={i} gap={8} marginVertical={4} paddingLeft={12}
                borderLeftWidth={2} borderLeftColor="$color9" opacity={0.85}>
                <InlineText text={block.text} />
              </XStack>
            );
          case 'ul':
            return (
              <YStack key={i} gap={2} marginVertical={2}>
                {block.items.map((item, j) => (
                  <XStack key={j} gap={6} paddingLeft={4}>
                    <Text fontSize={14} color="$color9" marginTop={1}>•</Text>
                    <InlineText text={item} />
                  </XStack>
                ))}
              </YStack>
            );
          case 'ol':
            return (
              <YStack key={i} gap={2} marginVertical={2}>
                {block.items.map((item, j) => (
                  <XStack key={j} gap={6} paddingLeft={4}>
                    <Text fontSize={14} color="$color10" fontWeight="600" minWidth={18}>
                      {j + 1}.
                    </Text>
                    <InlineText text={item} />
                  </XStack>
                ))}
              </YStack>
            );
          case 'hr':
            return (
              <YStack key={i} marginVertical={8}
                borderBottomWidth={0.5} borderBottomColor="$borderColor" />
            );
          case 'paragraph':
            return (
              <YStack key={i} marginVertical={2}>
                <InlineText text={block.text} />
              </YStack>
            );
          default:
            return null;
        }
      })}
    </YStack>
  );
}
