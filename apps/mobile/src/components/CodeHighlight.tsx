/**
 * Lightweight syntax highlighter for React Native.
 * Uses regex-based tokenization — no web dependencies.
 */

import React from 'react'
import { Text } from 'tamagui'
import { SYNTAX_COLORS } from '../theme/colors'

type Token = { text: string; color: string; bold?: boolean; italic?: boolean }

const KEYWORD_COLOR = SYNTAX_COLORS.keyword
const STRING_COLOR = SYNTAX_COLORS.string
const NUMBER_COLOR = SYNTAX_COLORS.number
const COMMENT_COLOR = SYNTAX_COLORS.comment
const FUNCTION_COLOR = SYNTAX_COLORS.function
const OPERATOR_COLOR = SYNTAX_COLORS.operator
const PUNCTUATION_COLOR = SYNTAX_COLORS.punctuation
const VARIABLE_COLOR = SYNTAX_COLORS.variable

const JS_KEYWORDS = new Set([
  'const',
  'let',
  'var',
  'function',
  'return',
  'if',
  'else',
  'for',
  'while',
  'do',
  'switch',
  'case',
  'break',
  'continue',
  'new',
  'this',
  'class',
  'extends',
  'import',
  'from',
  'export',
  'default',
  'async',
  'await',
  'try',
  'catch',
  'finally',
  'throw',
  'typeof',
  'instanceof',
  'in',
  'of',
  'true',
  'false',
  'null',
  'undefined',
  'void',
  'delete',
  'yield',
  'static',
  'super',
  'with',
  'debugger',
])

const PY_KEYWORDS = new Set([
  'def',
  'class',
  'return',
  'if',
  'elif',
  'else',
  'for',
  'while',
  'break',
  'continue',
  'import',
  'from',
  'as',
  'try',
  'except',
  'finally',
  'raise',
  'with',
  'yield',
  'lambda',
  'pass',
  'del',
  'global',
  'nonlocal',
  'assert',
  'True',
  'False',
  'None',
  'and',
  'or',
  'not',
  'in',
  'is',
  'async',
  'await',
])

const LANG_KEYWORDS: Record<string, Set<string>> = {
  javascript: JS_KEYWORDS,
  typescript: JS_KEYWORDS,
  jsx: JS_KEYWORDS,
  tsx: JS_KEYWORDS,
  python: PY_KEYWORDS,
  py: PY_KEYWORDS,
}

function tokenizeLine(line: string, lang: string): Token[] {
  const keywords = LANG_KEYWORDS[lang] || LANG_KEYWORDS.javascript
  const tokens: Token[] = []
  let i = 0

  while (i < line.length) {
    // Comments
    if (line[i] === '/' && line[i + 1] === '/') {
      tokens.push({ text: line.slice(i), color: COMMENT_COLOR, italic: true })
      break
    }
    if (
      line[i] === '#' &&
      (lang === 'python' || lang === 'py' || lang === 'yaml' || lang === 'bash' || lang === 'sh')
    ) {
      tokens.push({ text: line.slice(i), color: COMMENT_COLOR, italic: true })
      break
    }

    // Strings
    if (line[i] === '"' || line[i] === "'" || line[i] === '`') {
      const quote = line[i]
      let j = i + 1
      while (j < line.length && line[j] !== quote) {
        if (line[j] === '\\') j++
        j++
      }
      tokens.push({ text: line.slice(i, j + 1), color: STRING_COLOR })
      i = j + 1
      continue
    }

    // Numbers
    if (/[0-9]/.test(line[i]) && (i === 0 || /[\s(,=+\-*/<>!&|^~[\]{};:]/.test(line[i - 1]))) {
      let j = i
      while (j < line.length && /[0-9.xXa-fA-F_]/.test(line[j])) j++
      tokens.push({ text: line.slice(i, j), color: NUMBER_COLOR })
      i = j
      continue
    }

    // Words (identifiers / keywords)
    if (/[a-zA-Z_$]/.test(line[i])) {
      let j = i
      while (j < line.length && /[a-zA-Z0-9_$]/.test(line[j])) j++
      const word = line.slice(i, j)
      if (keywords.has(word)) {
        tokens.push({ text: word, color: KEYWORD_COLOR, bold: true })
      } else if (j < line.length && line[j] === '(') {
        tokens.push({ text: word, color: FUNCTION_COLOR })
      } else {
        tokens.push({ text: word, color: '#E0DAF0' })
      }
      i = j
      continue
    }

    // Operators
    if (/[+\-*/%=<>!&|^~?:]/.test(line[i])) {
      let j = i
      while (j < line.length && /[+\-*/%=<>!&|^~?:]/.test(line[j])) j++
      tokens.push({ text: line.slice(i, j), color: OPERATOR_COLOR })
      i = j
      continue
    }

    // Punctuation
    if (/[(){}[\];,.]/.test(line[i])) {
      tokens.push({ text: line[i], color: PUNCTUATION_COLOR })
      i++
      continue
    }

    // Whitespace / other
    tokens.push({ text: line[i], color: '#E0DAF0' })
    i++
  }

  return tokens
}

export function HighlightedCode({
  code,
  language,
  colors,
}: {
  code: string
  language: string
  colors: any
}) {
  const lines = code.split('\n')
  const lang = language.toLowerCase()

  return (
    <>
      {lines.map((line, lineIdx) => {
        if (!line.trim()) {
          return (
            <Text key={lineIdx} fontFamily="mono" fontSize={12} lineHeight={18}>
              {'\n'}
            </Text>
          )
        }
        const tokens = tokenizeLine(line, lang)
        return (
          <Text key={lineIdx} fontFamily="mono" fontSize={12} lineHeight={18}>
            {tokens.map((token, ti) => (
              <Text
                key={ti}
                color={token.color}
                fontWeight={token.bold ? '700' : '400'}
                fontStyle={token.italic ? 'italic' : 'normal'}
              >
                {token.text}
              </Text>
            ))}
            {lineIdx < lines.length - 1 ? '\n' : ''}
          </Text>
        )
      })}
    </>
  )
}
