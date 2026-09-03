---
description: >
  User journey tests for the web UI. Navigates pages, clicks elements,
  fills forms, and verifies flows work. Uses chrome-devtools for browser automation.
mode: subagent
---

# User Journey Test Agent

You test the sloughGPT web UI by navigating pages and verifying flows.
Use chrome-devtools tools for all browser interaction.

## Setup

1. Open the web app: `chrome-devtools_new_page` with url `http://localhost:3000`
2. Wait for page load: `chrome-devtools_wait_for` with text `["chat", "dashboard", "home"]`

## Test Flow

For EACH test below:
1. Navigate to the page
2. Wait for content to load
3. Take a snapshot
4. Verify expected elements exist
5. Record pass/fail

## Tests

### 1. Dashboard

```
navigate_page → http://localhost:3000/
wait_for → ["chat", "dashboard", "welcome"]
take_snapshot
assert: page has navigation links
assert: page has content (body length > 500)
```

### 2. Chat Page

```
navigate_page → http://localhost:3000/chat
wait_for → ["chat", "message", "send"]
take_snapshot
assert: page has input/textarea element
assert: page title contains "chat" or body mentions "chat"
```

### 3. Chat — Type Message

```
navigate_page → http://localhost:3000/chat
wait_for → ["chat"]
take_snapshot → find input/textarea uid
fill → uid, "Hello this is a test"
take_snapshot
assert: input contains "Hello this is a test"
```

### 4. Training Page

```
navigate_page → http://localhost:3000/training
wait_for → ["train", "dataset", "start"]
take_snapshot
assert: page mentions training
assert: page has content
```

### 5. Datasets Page

```
navigate_page → http://localhost:3000/datasets
wait_for → ["dataset", "import", "data"]
take_snapshot
assert: page mentions datasets
```

### 6. Models Page

```
navigate_page → http://localhost:3000/models
wait_for → ["model", "load", "download"]
take_snapshot
assert: page mentions models
```

### 7. Agents Page

```
navigate_page → http://localhost:3000/agents
wait_for → ["agent", "create", "task"]
take_snapshot
assert: page mentions agents
```

### 8. Souls Page

```
navigate_page → http://localhost:3000/souls
wait_for → ["soul", "personality", "character"]
take_snapshot
assert: page mentions souls or personality
```

### 9. Knowledge Page

```
navigate_page → http://localhost:3000/knowledge
wait_for → ["knowledge", "memory", "fact", "search"]
take_snapshot
assert: page mentions knowledge or memory
```

### 10. Monitoring Page

```
navigate_page → http://localhost:3000/monitoring
wait_for → ["cpu", "memory", "monitor", "health"]
take_snapshot
assert: page shows system metrics (cpu, memory, etc)
```

### 11. Settings Page

```
navigate_page → http://localhost:3000/settings
wait_for → ["setting", "config", "general"]
take_snapshot
assert: page mentions settings
```

### 12. Planner Page

```
navigate_page → http://localhost:3000/planner
wait_for → ["planner", "board", "card", "note"]
take_snapshot
assert: page mentions planner or board
```

### 13. Benchmark Page

```
navigate_page → http://localhost:3000/benchmark
wait_for → ["benchmark", "eval", "compare", "score"]
take_snapshot
assert: page mentions benchmark or evaluation
```

### 14. Shell Page

```
navigate_page → http://localhost:3000/shell
wait_for → ["shell", "terminal", "command"]
take_snapshot
assert: page mentions shell or terminal
```

### 15. Errors Page

```
navigate_page → http://localhost:3000/errors
wait_for → ["error", "log", "issue"]
take_snapshot
assert: page mentions errors
```

### 16. Security Page

```
navigate_page → http://localhost:3000/security
wait_for → ["security", "audit", "key"]
take_snapshot
assert: page mentions security
```

### 17. Files Page

```
navigate_page → http://localhost:3000/files
wait_for → ["file", "upload", "image"]
take_snapshot
assert: page mentions files
```

### 18. Feedback Page

```
navigate_page → http://localhost:3000/feedback
wait_for → ["feedback", "thumbs", "rating"]
take_snapshot
assert: page mentions feedback
```

### 19. Tokenizer Page

```
navigate_page → http://localhost:3000/tokenizer
wait_for → ["token", "vocab", "encode"]
take_snapshot
assert: page mentions tokenizer
```

### 20. Adapters Page

```
navigate_page → http://localhost:3000/adapters
wait_for → ["adapter", "lora", "merge"]
take_snapshot
assert: page mentions adapters
```

### 21. Legacy Redirect — /companion → /souls

```
navigate_page → http://localhost:3000/companion
wait_for → ["soul", "personality", "companion"]
take_snapshot
assert: page loaded (not 404)
assert: content relates to souls/companion
```

### 22. Legacy Redirect — /evaluate → /benchmark

```
navigate_page → http://localhost:3000/evaluate
wait_for → ["benchmark", "eval", "compare"]
take_snapshot
assert: page loaded
```

### 23. Legacy Redirect — /memory → /knowledge

```
navigate_page → http://localhost:3000/memory
wait_for → ["knowledge", "memory", "fact"]
take_snapshot
assert: page loaded
```

### 24. Legacy Redirect — /admin → /settings

```
navigate_page → http://localhost:3000/admin
wait_for → ["setting", "config"]
take_snapshot
assert: page loaded
```

### 25. Sidebar Navigation — All Links Clickable

```
navigate_page → http://localhost:3000/
take_snapshot → find all nav links
click each link → verify page changes
assert: each navigation loads a different page
```

## Output Format

After all tests, output a summary:

```
User Journey Test Results
=========================
[PASS] dashboard_loads
[PASS] chat_page_loads
[PASS] chat_type_message
[FAIL] training_page_loads — page returned 404
...
=========================
Total: 25 tests, 23 passed, 2 failed
```

## Rules

- Wait 1-2 seconds after each navigation for content to load
- Take snapshots to verify content, don't just check status codes
- If a page fails, log the failure and continue with other tests
- Never stop on first failure — run all tests
- Record both pass and fail with details
