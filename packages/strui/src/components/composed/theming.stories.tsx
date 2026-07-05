import type { Meta, StoryObj } from '@storybook/react'
import { useState } from 'react'
import { Button } from '../ui/button'
import { NavRail, NavRailLink } from './nav-rail'
import { DEFAULT_THEME_SWATCHES, type ThemeSwatch } from './theme-color-picker'

const meta = {
  title: 'Composed/Theming',
  parameters: {
    layout: 'fullscreen',
  },
} satisfies Meta<{ themeColor: string }>

export default meta
type Story = StoryObj<typeof meta>

export const LiveThemingDemo: Story = {
  render: () => {
    const [themeColor, setThemeColor] = useState('#a67fd4')

    return (
      <div
        style={{
          '--primary': themeColor,
          '--ring': themeColor,
        } as React.CSSProperties}
        className="min-h-screen bg-background p-6"
      >
        <div className="mb-8 space-y-4">
          <h2 className="sl-h2">Live Theme Preview</h2>
          <p className="sl-muted">
            Click any swatch to see hover states update. All components use sl-hover-primary for themed hover.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <span className="sl-label mr-2">Accent:</span>
            {DEFAULT_THEME_SWATCHES.map((swatch) => (
              <button
                key={swatch.id}
                onClick={() => setThemeColor(swatch.color)}
                className={`sl-hover-primary h-10 w-10 rounded-none transition-all ${
                  themeColor === swatch.color ? 'ring-2 ring-ring ring-offset-2' : ''
                }`}
                style={{ backgroundColor: swatch.color }}
                title={swatch.name}
              />
            ))}
          </div>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div className="space-y-4">
            <h3 className="sl-h2">Buttons</h3>
            <div className="flex flex-wrap gap-3">
              <Button>Primary</Button>
              <Button variant="secondary">Secondary</Button>
              <Button variant="outline">Outline</Button>
              <Button variant="ghost">Ghost</Button>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="sl-h2">Nav Rail</h3>
            <NavRail className="max-w-[200px]">
              <NavRailLink href="#">Home</NavRailLink>
              <NavRailLink href="#" active>
                Dashboard
              </NavRailLink>
              <NavRailLink href="#">Settings</NavRailLink>
            </NavRail>
          </div>
        </div>
      </div>
    )
  },
}

export const HoverStates: Story = {
  render: () => {
    const [themeColor, setThemeColor] = useState('#a67fd4')

    return (
      <div
        style={{
          '--primary': themeColor,
          '--ring': themeColor,
        } as React.CSSProperties}
        className="min-h-screen bg-background p-6"
      >
        <div className="mb-6 flex items-center gap-4">
          <span className="sl-label">Theme:</span>
          <input
            type="color"
            value={themeColor}
            onChange={(e) => setThemeColor(e.target.value)}
            className="h-10 w-14 cursor-pointer rounded-none border border-border"
          />
          <span className="font-mono text-sm">{themeColor}</span>
        </div>

        <div className="space-y-6">
          <div className="space-y-2">
            <h3 className="sl-h2">Hover over any element</h3>
            <p className="sl-muted">
              Notice the hover tint matches the chosen theme color
            </p>
          </div>

          <div className="flex flex-wrap gap-3 rounded-none border border-border bg-card p-4">
            <Button variant="secondary">Secondary</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="ghost">Ghost</Button>
          </div>

          <NavRail className="max-w-[240px]">
            <NavRailLink href="#">Item One</NavRailLink>
            <NavRailLink href="#">Item Two</NavRailLink>
            <NavRailLink href="#">Item Three</NavRailLink>
          </NavRail>
        </div>
      </div>
    )
  },
}
