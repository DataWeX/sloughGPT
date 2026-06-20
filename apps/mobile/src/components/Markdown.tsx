import React from 'react';
import {Text, StyleSheet} from 'react-native';
import {colors, spacing, radii, typography} from '../theme';

interface Props {
  content: string;
  style?: any;
}

export function Markdown({content, style}: Props) {
  if (!content) return null;

  const parts = content.split(/(```[\s\S]*?```|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|#{1,3} .+|\n)/g);

  return (
    <Text style={[styles.text, style]}>
      {parts.map((part, i) => {
        if (part.startsWith('```') && part.endsWith('```')) {
          const code = part.slice(3, -3).replace(/^\w+\n/, '');
          return (
            <Text key={i} style={styles.codeBlock}>
              {code}
            </Text>
          );
        }
        if (part.startsWith('`') && part.endsWith('`')) {
          return (
            <Text key={i} style={styles.code}>
              {part.slice(1, -1)}
            </Text>
          );
        }
        if (part.startsWith('**') && part.endsWith('**')) {
          return (
            <Text key={i} style={styles.bold}>
              {part.slice(2, -2)}
            </Text>
          );
        }
        if (part.startsWith('*') && part.endsWith('*') && !part.startsWith('**')) {
          return (
            <Text key={i} style={styles.italic}>
              {part.slice(1, -1)}
            </Text>
          );
        }
        if (part.startsWith('# ')) {
          return (
            <Text key={i} style={styles.h1}>
              {part.slice(2)}
            </Text>
          );
        }
        if (part.startsWith('## ')) {
          return (
            <Text key={i} style={styles.h2}>
              {part.slice(3)}
            </Text>
          );
        }
        if (part.startsWith('### ')) {
          return (
            <Text key={i} style={styles.h3}>
              {part.slice(4)}
            </Text>
          );
        }
        if (part === '\n') {
          return <Text key={i}>{'\n'}</Text>;
        }
        return <Text key={i}>{part}</Text>;
      })}
    </Text>
  );
}

const styles = StyleSheet.create({
  text: {
    ...typography.body,
  },
  bold: {
    fontWeight: '700',
  },
  italic: {
    fontStyle: 'italic',
  },
  code: {
    ...typography.mono,
    backgroundColor: colors.surface,
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderRadius: 3,
    color: colors.primary,
  },
  codeBlock: {
    ...typography.mono,
    backgroundColor: colors.surface,
    padding: spacing.md,
    borderRadius: radii.sm,
    marginTop: 4,
    marginBottom: 4,
    color: colors.text,
  },
  h1: {
    ...typography.h1,
    marginTop: 8,
    marginBottom: 4,
  },
  h2: {
    ...typography.h2,
    marginTop: 8,
    marginBottom: 4,
  },
  h3: {
    ...typography.h3,
    marginTop: 6,
    marginBottom: 2,
  },
});
