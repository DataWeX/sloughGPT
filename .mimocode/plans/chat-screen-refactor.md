# Plan: Chat Screen Refactor — Frontend Engineering Principles

## Goal
Refactor the chat screen to follow frontend engineering best practices: proper hook decomposition, eliminate dual state tracking, optimize streaming performance, and fix architectural anti-patterns.

## Issues Found (Priority Order)

### P0 — Critical
1. **`useChatMessages` is 653 lines** — does too much (session CRUD, streaming, images, feedback, search)
2. **`messagesRef` + `messages` dual-tracking** — manually synced refs alongside state, fragile and bug-prone
3. **`sendMessage` not wrapped in useCallback** — passed as prop, causes child re-renders on every render
4. **Duplicate PDFUpload** in `ChatInputAccessories` — rendered twice (lines 52-56 and 73-75)

### P1 — Performance
5. **No message virtualization** — all messages rendered in DOM, long conversations cause layout thrash
6. **Markdown re-parses on every streaming token** — full parse on each token for long messages
7. **MoodOrb uses `setInterval(4000)`** — 25 state updates/sec for cosmetic animation, should use CSS
8. **`computeSearchMatches` runs when search is empty** — iterates all messages for zero results

### P2 — Architecture
9. **ChatContext value is 40+ fields** — causes broad re-renders across all consumers
10. **Session save `useEffect` has stale dependency** — `saveSession` recreates on every `messages` change, resetting the debounce timeout
11. **`handleRegenerate` dependency array incomplete** — missing `messages` (uses `messagesRef` instead)
12. **Raw `fetch()` calls in page effects** — bypass centralized HTTP client

## Files to Modify

| File | Change |
|------|--------|
| `hooks/useChatMessages.ts` | Split into 3 focused hooks |
| `hooks/useChatStreaming.ts` | **NEW** — extracted streaming logic |
| `hooks/useChatSessions.ts` | **NEW** — extracted session CRUD |
| `components/chat/ChatInputAccessories.tsx` | Remove duplicate PDFUpload |
| `components/chat/EmptyState.tsx` | Replace setInterval with CSS animation |
| `components/chat/ChatArea.tsx` | Add message virtualization |
| `components/chat/ChatScreen.tsx` | Optimize filtered messages computation |
| `app/(app)/chat/page.tsx` | Wire new hooks, reduce context blast radius |
| `lib/chat-utils.ts` | Fix `exportConversationAsMarkdown` to actually download |

## Implementation Steps

### Step 1: Extract `useChatSessions` hook
Extract session CRUD (load, delete, star, pin, rename, duplicate, merge) from `useChatMessages` into a new `useChatSessions` hook. This hook owns `sessions` state and all session mutations.

### Step 2: Extract `useChatStreaming` hook  
Extract the core `sendMessage` function and streaming logic into `useChatStreaming`. This hook owns `messages`, `loading`, `input`, `images` state and all streaming callbacks. It receives config from the parent and exposes clean callbacks.

### Step 3: Fix dual-tracking anti-pattern
Replace `messagesRef.current = updated` pattern inside `setMessages` callbacks with a single source of truth. Use a `useRef` that's only read inside callbacks (not rendered), and update it in a `useEffect` that watches `messages`. This eliminates the fragile manual sync.

### Step 4: Wrap `sendMessage` in useCallback
The core send function must be stable across renders to prevent child re-renders. Wrap in `useCallback` with proper dependency management.

### Step 5: Fix duplicate PDFUpload
Remove one of the two `<PDFUpload>` renders in `ChatInputAccessories.tsx`.

### Step 6: Replace MoodOrb setInterval with CSS
Replace the `setInterval` state updates with a pure CSS animation (`@keyframes` for mood rotation). No React state needed.

### Step 7: Short-circuit `computeSearchMatches`
Add early return when `searchQuery` is empty to avoid iterating all messages.

### Step 8: Optimize ChatContext value
Split the monolithic context into 2-3 focused contexts: `ChatMessagesContext`, `ChatModelContext`, `ChatUIContext`. This reduces re-render blast radius.

### Step 9: Fix session save debounce
Use a ref-based debounce that doesn't recreate on every `messages` change.

### Step 10: Fix `exportConversationAsMarkdown`
Add actual file download (create Blob → URL.createObjectURL → click).

## Verification
1. `npx tsc --noEmit` — 0 errors
2. `npx vitest run` — all tests pass
3. Manual: send messages, verify streaming works
4. Manual: load session, verify messages appear
5. Manual: search messages, verify highlight works
6. Manual: regenerate last message
7. Manual: edit user message
8. Manual: thumbs up/down feedback
9. Manual: empty state shows suggestions
10. Manual: tool panel tabs work
