/**
 * Color, type, spacing and radius token tables for the brand guide.
 * Pure data — no React, no IO.
 */

/* ──────────────────────────────────────────────────────────────────
 * Color token definitions — light & dark mode RGB triples
 * ────────────────────────────────────────────────────────────────── */

export const lightColors = [
  { name: '--background', rgb: '248 246 252', label: 'Background', desc: 'Page background' },
  { name: '--foreground', rgb: '25 22 36', label: 'Foreground', desc: 'Primary text' },
  { name: '--card', rgb: '255 255 255', label: 'Card', desc: 'Card/panel backgrounds' },
  { name: '--card-foreground', rgb: '25 22 36', label: 'Card FG', desc: 'Text on cards' },
  { name: '--primary', rgb: '124 82 196', label: 'Primary', desc: 'Buttons, links, active' },
  {
    name: '--primary-foreground',
    rgb: '250 248 255',
    label: 'Primary FG',
    desc: 'Text on primary',
  },
  { name: '--secondary', rgb: '237 232 248', label: 'Secondary', desc: 'Secondary backgrounds' },
  {
    name: '--secondary-foreground',
    rgb: '42 37 55',
    label: 'Secondary FG',
    desc: 'Text on secondary',
  },
  { name: '--muted', rgb: '244 242 248', label: 'Muted', desc: 'Subtle backgrounds' },
  {
    name: '--muted-foreground',
    rgb: '130 122 150',
    label: 'Muted FG',
    desc: 'Captions, secondary',
  },
  { name: '--accent', rgb: '236 145 95', label: 'Accent', desc: 'Highlights, warnings' },
  { name: '--accent-foreground', rgb: '250 248 255', label: 'Accent FG', desc: 'Text on accent' },
  { name: '--border', rgb: '228 224 242', label: 'Border', desc: 'Borders, dividers' },
  { name: '--input', rgb: '228 224 242', label: 'Input', desc: 'Input borders' },
  { name: '--ring', rgb: '124 82 196', label: 'Ring', desc: 'Focus rings' },
  { name: '--success', rgb: '52 176 125', label: 'Success', desc: 'Success states' },
  { name: '--warning', rgb: '236 168 60', label: 'Warning', desc: 'Warning states' },
  { name: '--destructive', rgb: '220 80 90', label: 'Destructive', desc: 'Errors, destructive' },
] as const

export const darkColors = [
  { name: '--background', rgb: '17 15 24', label: 'Background', desc: 'Deep charcoal-violet' },
  { name: '--foreground', rgb: '238 234 248', label: 'Foreground', desc: 'Primary text' },
  { name: '--card', rgb: '28 25 38', label: 'Card', desc: 'Card/panel backgrounds' },
  { name: '--card-foreground', rgb: '238 234 248', label: 'Card FG', desc: 'Text on cards' },
  { name: '--primary', rgb: '192 170 244', label: 'Primary', desc: 'Lilac primary' },
  { name: '--primary-foreground', rgb: '25 22 36', label: 'Primary FG', desc: 'Text on primary' },
  { name: '--secondary', rgb: '50 44 68', label: 'Secondary', desc: 'Secondary backgrounds' },
  {
    name: '--secondary-foreground',
    rgb: '238 234 248',
    label: 'Secondary FG',
    desc: 'Text on secondary',
  },
  { name: '--muted', rgb: '38 34 52', label: 'Muted', desc: 'Subtle backgrounds' },
  {
    name: '--muted-foreground',
    rgb: '150 140 172',
    label: 'Muted FG',
    desc: 'Captions, secondary',
  },
  { name: '--accent', rgb: '240 176 130', label: 'Accent', desc: 'Warm peach accent' },
  { name: '--accent-foreground', rgb: '25 22 36', label: 'Accent FG', desc: 'Text on accent' },
  { name: '--border', rgb: '52 46 72', label: 'Border', desc: 'Borders, dividers' },
  { name: '--input', rgb: '52 46 72', label: 'Input', desc: 'Input borders' },
  { name: '--ring', rgb: '192 170 244', label: 'Ring', desc: 'Focus rings' },
  { name: '--success', rgb: '72 192 140', label: 'Success', desc: 'Success states' },
  { name: '--warning', rgb: '240 192 80', label: 'Warning', desc: 'Warning states' },
  { name: '--destructive', rgb: '235 100 110', label: 'Destructive', desc: 'Errors, destructive' },
] as const

export const accentThemes = [
  { name: 'blue', label: 'Blue', rgb: '90 130 220', desc: 'Calm ocean, trustworthy' },
  { name: 'purple', label: 'Purple', rgb: '155 108 214', desc: 'Ethereal lilac, mystical' },
  { name: 'pink', label: 'Pink', rgb: '218 130 170', desc: 'Soft rose, delicate' },
  { name: 'red', label: 'Red', rgb: '230 120 130', desc: 'Warm coral, energetic' },
  { name: 'orange', label: 'Orange', rgb: '236 155 90', desc: 'Peachy bread crust, warm bakery' },
  { name: 'green', label: 'Green', rgb: '72 178 130', desc: 'Fresh mint, natural' },
  { name: 'teal', label: 'Teal', rgb: '72 166 200', desc: 'Cool dew, refreshing' },
] as const

export const typographyScale = [
  {
    role: 'Page Title',
    class: 'text-2xl md:text-3xl font-semibold',
    sample: 'Noir Violet',
    font: 'font-[family-name:var(--font-rubik)]',
    weight: '600',
  },
  {
    role: 'Section Title',
    class: 'text-[11px] font-medium',
    sample: 'Design System',
    font: 'font-[family-name:var(--font-rubik)]',
    weight: '500',
  },
  {
    role: 'Body',
    class: 'text-sm',
    sample: 'Warm, sophisticated, technical.',
    font: 'font-[family-name:var(--font-rubik)]',
    weight: '400',
  },
  {
    role: 'Caption',
    class: 'text-xs text-muted-foreground',
    sample: 'Last updated 2 minutes ago',
    font: 'font-[family-name:var(--font-rubik)]',
    weight: '400',
  },
  {
    role: 'Label',
    class: 'text-xs font-medium uppercase tracking-wider',
    sample: 'STATUS',
    font: 'font-[family-name:var(--font-rubik)]',
    weight: '500',
  },
  {
    role: 'Badge',
    class: 'text-[10px] font-medium',
    sample: 'Active',
    font: 'font-[family-name:var(--font-rubik)]',
    weight: '500',
  },
  {
    role: 'Mono',
    class: 'font-mono text-xs',
    sample: 'rgb(124, 82, 196)',
    font: 'font-[family-name:var(--font-jetbrains-mono)]',
    weight: '400',
  },
] as const

export const spacingScale = [
  { token: 'gap-1', px: '4px', class: 'gap-1' },
  { token: 'gap-2', px: '8px', class: 'gap-2' },
  { token: 'gap-3', px: '12px', class: 'gap-3' },
  { token: 'gap-4', px: '16px', class: 'gap-4' },
  { token: 'gap-6', px: '24px', class: 'gap-6' },
]

export const radiusScale = [
  { token: 'rounded-none', px: '0px' },
  { token: 'rounded-sm', px: '2px' },
  { token: 'rounded', px: '4px' },
  { token: 'rounded-md', px: '6px' },
  { token: 'rounded-lg', px: '10px' },
  { token: 'rounded-xl', px: '14px' },
]
