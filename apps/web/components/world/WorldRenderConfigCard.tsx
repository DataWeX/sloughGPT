'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input, Label } from '@sloughgpt/strui'

interface RenderConfig {
  width?: number
  height?: number
  samples?: number
  camera_height?: number
  camera_distance?: number
}

interface WorldRenderConfigCardProps {
  config: RenderConfig
  onConfigChange: (config: RenderConfig) => void
  onRender: () => void
  onTick?: (neural?: boolean) => void
  rendering?: boolean
  ticking?: boolean
}

export function WorldRenderConfigCard({
  config,
  onConfigChange,
  onRender,
  onTick,
  rendering = false,
  ticking = false,
}: WorldRenderConfigCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Render Config</CardTitle>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5 space-y-4">
        <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-5">
          <div className="space-y-1">
            <Label className="text-xs">Width</Label>
            <Input
              type="number"
              value={config.width ?? 160}
              onChange={(e) => onConfigChange({ ...config, width: Number(e.target.value) })}
              className="h-7 text-[11px]"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Height</Label>
            <Input
              type="number"
              value={config.height ?? 120}
              onChange={(e) => onConfigChange({ ...config, height: Number(e.target.value) })}
              className="h-7 text-[11px]"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Samples</Label>
            <Input
              type="number"
              value={config.samples ?? 16}
              onChange={(e) => onConfigChange({ ...config, samples: Number(e.target.value) })}
              className="h-7 text-[11px]"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Camera Height</Label>
            <Input
              type="number"
              step="0.5"
              value={config.camera_height ?? 40}
              onChange={(e) => onConfigChange({ ...config, camera_height: Number(e.target.value) })}
              className="h-7 text-[11px]"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Camera Distance</Label>
            <Input
              type="number"
              step="0.5"
              value={config.camera_distance ?? 30}
              onChange={(e) => onConfigChange({ ...config, camera_distance: Number(e.target.value) })}
              className="h-7 text-[11px]"
            />
          </div>
        </div>
        <div className="flex gap-2">
          <Button onClick={onRender} disabled={rendering} className="flex-1 h-7 text-[11px]">
            {rendering ? 'Rendering...' : 'Render'}
          </Button>
          {onTick && (
            <>
              <Button onClick={() => onTick(false)} disabled={ticking} variant="outline" className="flex-1 h-7 text-[11px]">
                {ticking ? 'Ticking...' : 'Run Tick'}
              </Button>
              <Button onClick={() => onTick(true)} disabled={ticking} variant="outline" className="flex-1 h-7 text-[11px]">
                {ticking ? 'Processing...' : 'Tick + Neural'}
              </Button>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
