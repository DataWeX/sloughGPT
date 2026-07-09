import React from 'react';
import {Text} from 'tamagui';

interface Props {
  content: string;
  style?: any;
}

export function Markdown({content, style}: Props) {
  if (!content) return null;

  const parts = content.split(/(```[\s\S]*?```|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|#{1,3} .+|\n)/g);

  return (
    <Text fontSize={15} fontWeight="400" color="$color" {...(style ? {style} : {})}>
      {parts.map((part, i) => {
        if (part.startsWith('```') && part.endsWith('```')) {
          const code = part.slice(3, -3).replace(/^\w+\n/, '');
          return (
            <Text key={i} fontFamily="mono" backgroundColor="$background" padding={12} borderRadius={4} marginTop={4} marginBottom={4} color="$color">
              {code}
            </Text>
          );
        }
        if (part.startsWith('`') && part.endsWith('`')) {
          return (
            <Text key={i} fontFamily="mono" backgroundColor="$background" paddingHorizontal={4} paddingVertical={1} borderRadius={3} color="$color9">
              {part.slice(1, -1)}
            </Text>
          );
        }
        if (part.startsWith('**') && part.endsWith('**')) {
          return (
            <Text key={i} fontWeight="700">
              {part.slice(2, -2)}
            </Text>
          );
        }
        if (part.startsWith('*') && part.endsWith('*') && !part.startsWith('**')) {
          return (
            <Text key={i} fontStyle="italic">
              {part.slice(1, -1)}
            </Text>
          );
        }
        if (part.startsWith('# ')) {
          return (
            <Text key={i} fontSize={20} fontWeight="700" letterSpacing={-0.3} marginTop={8} marginBottom={4}>
              {part.slice(2)}
            </Text>
          );
        }
        if (part.startsWith('## ')) {
          return (
            <Text key={i} fontSize={18} fontWeight="600" letterSpacing={-0.2} marginTop={8} marginBottom={4}>
              {part.slice(3)}
            </Text>
          );
        }
        if (part.startsWith('### ')) {
          return (
            <Text key={i} fontSize={16} fontWeight="600" marginTop={6} marginBottom={2}>
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
