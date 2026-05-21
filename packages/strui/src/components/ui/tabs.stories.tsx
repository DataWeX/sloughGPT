import type { Meta, StoryObj } from '@storybook/react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './tabs'

const meta = {
  title: 'UI/Tabs',
  tags: ['autodocs'],
} satisfies Meta

export default meta

export const Default: StoryObj = {
  render: () => (
    <Tabs defaultValue="params" className="w-full max-w-md">
      <TabsList className="w-full justify-start">
        <TabsTrigger value="params" className="str-touch-target min-w-[5rem]">
          Params
        </TabsTrigger>
        <TabsTrigger value="logs" className="str-touch-target min-w-[5rem]">
          Logs
        </TabsTrigger>
      </TabsList>
      <TabsContent value="params" className="text-sm text-muted-foreground">
        Temperature, top-p, and max tokens for this session.
      </TabsContent>
      <TabsContent value="logs" className="font-mono text-xs text-muted-foreground">
        SYS.LOG — streaming output would appear here.
      </TabsContent>
    </Tabs>
  ),
}
