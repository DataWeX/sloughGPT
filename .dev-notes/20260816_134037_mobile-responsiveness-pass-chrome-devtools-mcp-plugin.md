---
id: 20260816_134037_mobile-responsiveness-pass-chrome-devtools-mcp-plugin
title: Mobile responsiveness pass + chrome-devtools-mcp plugin
status: done
tags: web,mobile,devtools,opencode
created: 2026-08-16T13:40:37.096869+00:00
---

Mobile responsiveness pass + chrome-devtools-mcp plugin

Installed chrome-devtools-mcp MCP server (opencode.json mcp.chrome-devtools) + /devtools command so devtools can be opened against the web app (desktop Chrome --remote-debugging-port=9222, or Android via adb reverse tcp:9222). Ran a phone-viewport (375px) audit of apps/web and applied targeted fixes: vm Execution Trace table + QuantizationCard table + KeyboardShortcutsModal table get overflow-x-auto; ImageUpload remove button and ConversationSidebar pin/star/unread/export/duplicate/archive/delete buttons enlarged to h-7 w-7 and made always-visible on mobile (sm:opacity-0 hover reveal kept on desktop); ChatInputRow now flex-wrap so accessories wrap instead of squeezing the textarea to ~60px; ChatSendButton h-11; ChatSearchBar match buttons h-7 w-7; errors page header flex-wrap + search w-full sm:w-48; model/[id] and dataset/[id] primary CTAs h-8 to h-11; ConfigureStep/DataStep/benchmark Quality grid-cols-3 to grid-cols-1 sm:grid-cols-3; StatusBar h-7 sm:h-8. Verified: tsc 0 errors, 29 chat/component test files (365 tests) + 6 page files (131 tests) all pass.