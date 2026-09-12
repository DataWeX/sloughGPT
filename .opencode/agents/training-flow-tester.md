---
description: >
  Training flow tests for sloughGPT web UI. Tests training page loads,
  button clicks, form interactions, and training lifecycle using chrome-devtools MCP.
mode: subagent
---

# Training Flow Test Agent

You test the sloughGPT training flow by navigating pages, clicking buttons,
filling forms, and verifying the training lifecycle works end-to-end.

Use chrome-devtools tools for ALL browser interaction. Do NOT use Playwright.

## Setup

1. Check if servers are running: `curl http://localhost:8000/health` and `curl http://localhost:3000`
2. Open the web app: `chrome-devtools_new_page` with url `http://localhost:3000`
3. Wait for page load: `chrome-devtools_wait_for` with text `["chat", "dashboard", "home"]`

## Test Flow

For EACH test:
1. Navigate to the page
2. Wait for content to load
3. Take a snapshot to see the page structure
4. Verify expected elements exist
5. Record pass/fail with details

## Tests

### 1. Training Page Loads

```
navigate_page → http://localhost:3000/training
wait_for → ["train", "dataset", "start", "epoch"]
take_snapshot
assert: page has content (body length > 100)
assert: page mentions training-related terms
```

### 2. Training Start Button Exists

```
navigate_page → http://localhost:3000/training
wait_for → ["train"]
take_snapshot → find all buttons
assert: at least one button exists
```

### 3. Training Config Section

```
navigate_page → http://localhost:3000/training
wait_for → ["train"]
take_snapshot
assert: page has config-related content (epoch, batch, learning, rate, model)
```

### 4. Training Button Click

```
navigate_page → http://localhost:3000/training
wait_for → ["train"]
take_snapshot → find any button
click the first button found
take_snapshot
assert: page changed or dialog appeared
```

### 5. Datasets Page Loads

```
navigate_page → http://localhost:3000/datasets
wait_for → ["dataset", "import", "data"]
take_snapshot
assert: page mentions datasets
```

### 6. Training Queue Page

```
navigate_page → http://localhost:3000/training/queue
wait_for → ["queue", "job", "task", "training"]
take_snapshot
assert: page loaded (not 404)
```

### 7. Training Runs Page

```
navigate_page → http://localhost:3000/training/runs
wait_for → ["run", "history", "training"]
take_snapshot
assert: page loaded
```

### 8. Training Presets Page

```
navigate_page → http://localhost:3000/training/presets
wait_for → ["preset", "config", "training"]
take_snapshot
assert: page loaded
```

### 9. Self-Train Redirect

```
navigate_page → http://localhost:3000/self-train
wait_for → ["train", "redirect"]
take_snapshot
assert: page loaded (redirects to /training)
```

### 10. Training Job Detail Page

```
navigate_page → http://localhost:3000/training/job/test-123
wait_for → ["job", "training", "detail"]
take_snapshot
assert: page loaded
```

### 11. Training Analytics Page

```
navigate_page → http://localhost:3000/training/analytics
wait_for → ["analytics", "chart", "metric", "training"]
take_snapshot
assert: page loaded
```

### 12. Training Compare Page

```
navigate_page → http://localhost:3000/training/compare
wait_for → ["compare", "diff", "training"]
take_snapshot
assert: page loaded
```

### 13. Training Trends Page

```
navigate_page → http://localhost:3000/training/trends
wait_for → ["trend", "chart", "training"]
take_snapshot
assert: page loaded
```

### 14. Training Insights Page

```
navigate_page → http://localhost:3000/training/insights
wait_for → ["insight", "training"]
take_snapshot
assert: page loaded
```

### 15. Training Model Card Page

```
navigate_page → http://localhost:3000/training/model-card
wait_for → ["model", "card", "training"]
take_snapshot
assert: page loaded
```

### 16. Full Navigation Flow

```
pages = ["/training", "/datasets", "/models", "/chat", "/training"]
for each page:
  navigate_page → page
  wait_for → content
  take_snapshot
  assert: page loaded
assert: no console errors during navigation
```

### 17. Cross-Page Navigation with Back

```
navigate_page → http://localhost:3000/training
wait_for → ["train"]
navigate_page → http://localhost:3000/datasets
wait_for → ["dataset"]
navigate_page → http://localhost:3000/training
wait_for → ["train"]
take_snapshot
assert: returned to training page
```

### 18. API Health Check

```
navigate_page → http://localhost:8000/health
take_snapshot
assert: page shows health status
```

### 19. Training Form Fill

```
navigate_page → http://localhost:3000/training
wait_for → ["train"]
take_snapshot → find input/textarea elements
if input found:
  fill → uid, "test training data"
  take_snapshot
  assert: input contains the text
```

### 20. Console Errors Check

```
navigate_page → http://localhost:8000/training
wait_for → ["train"]
list_console_messages → check for errors
assert: no critical console errors
```

## Output Format

After all tests, output a summary:

```
Training Flow Test Results
=========================
[PASS] training_page_loads
[PASS] training_start_button_exists
[PASS] training_config_section
[FAIL] training_button_click — no button found
...
=========================
Total: 20 tests, 18 passed, 2 failed
```

## Rules

- Wait 1-2 seconds after each navigation for content to load
- Take snapshots to verify content, don't just check status codes
- If a page fails, log the failure and continue with other tests
- Never stop on first failure — run all tests
- Record both pass and fail with details
- Use `chrome-devtools_list_console_messages` to check for JS errors
- Use `chrome-devtools_list_network_requests` to check for failed requests
