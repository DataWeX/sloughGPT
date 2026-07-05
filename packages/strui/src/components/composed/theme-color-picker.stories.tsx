import type { Meta, StoryObj } from '@storybook/react'
import { useState } from 'react'
import { ThemeColorPicker, ColorInput, DEFAULT_THEME_SWATCHES, type ThemeSwatch } from './theme-color-picker'

const meta = {
  title: 'Composed/ThemeColorPicker',
  component: ThemeColorPicker,
  parameters: {
    layout: 'padded',
  },
  args: {
    value: '#a67fd4',
    onChange: () => {},
  },
} satisfies Meta<typeof ThemeColorPicker>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: (args) => {
    const [color, setColor] = useState(args.value)
    return <ThemeColorPicker {...args} value={color} onChange={setColor} />
  },
}

export const WithCustomColor: Story = {
  render: (args) => {
    const [color, setColor] = useState(args.value)
    return (
      <ThemeColorPicker
        {...args}
        value={color}
        onChange={setColor}
        onCustomColor={(c) => console.log('Custom color:', c)}
        showCustomInput
        label="Choose accent color"
      />
    )
  },
}

export const SwatchWithLabels: Story = {
  render: () => {
    const [color, setColor] = useState('#a67fd4')
    return (
      <div className="space-y-4">
        <p className="sl-muted">Preset swatches</p>
        <div className="flex flex-wrap gap-1">
          {DEFAULT_THEME_SWATCHES.map((swatch) => (
            <button
              key={swatch.id}
              onClick={() => setColor(swatch.color)}
              className={`sl-hover-primary flex items-center gap-2 px-3 py-2 rounded-none transition-all ${
                color === swatch.color ? 'ring-2 ring-ring' : ''
              }`}
            >
              <span
                className="h-5 w-5 rounded-none"
                style={{ backgroundColor: swatch.color }}
              />
              <span className="text-sm">{swatch.name}</span>
            </button>
          ))}
        </div>
      </div>
    )
  },
}

export const ColorInputOnly: Story = {
  render: () => {
    const [color, setColor] = useState('#a67fd4')
    return <ColorInput value={color} onChange={(e) => setColor(e.target.value)} label="Pick a color" />
  },
}
