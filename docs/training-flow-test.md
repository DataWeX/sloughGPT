# Training Flow Quick Test

Run this test using opencode's chrome-devtools MCP.

## Prerequisites

1. Start the API server: `FORCE_COLOR=1 ./sloughgpt serve` (port 8000)
2. Start the web app: `cd apps/web && npm run dev` (port 3000)
3. Run in opencode: `/test-training-flow`

## Manual Test Steps

### Step 1: Open Training Page
```
chrome-devtools_new_page → http://localhost:3000/training
```

### Step 2: Wait for Load
```
chrome-devtools_wait_for → ["train", "dataset", "start"]
```

### Step 3: Take Snapshot
```
chrome-devtools_take_snapshot
```

### Step 4: Check for Buttons
Look for any button elements in the snapshot output.

### Step 5: Click Training Button
```
chrome-devtools_click → pageId=<id>, uid=<button_uid>
```

### Step 6: Verify Result
```
chrome-devtools_take_snapshot
```

### Step 7: Check Console Errors
```
chrome-devtools_list_console_messages → types=["error"]
```

### Step 8: Check Network Errors
```
chrome-devtools_list_network_requests → resourceTypes=["xhr", "fetch"]
```

## Expected Results

- Training page loads with content
- At least one button exists
- No critical console errors
- No failed network requests (status >= 400)
