#!/bin/bash
# Start the SloughGPT API server with unified logging
# Uses Xcode Python which has torch installed

cd /Users/mac/sloughGPT

echo "Starting SloughGPT API server..."
echo "Using Python: /usr/bin/python3"

# Start unified log server first
nohup /usr/bin/python3 scripts/unified-log-server.py > /tmp/sloughgpt-log-server.log 2>&1 &
LOG_PID=$!
sleep 1

send_log() {
    curl -s -X POST http://localhost:9999/log \
        -H "Content-Type: application/json" \
        -d "{\"source\":\"api\",\"level\":\"$1\",\"message\":\"$2\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)\"}" > /dev/null 2>&1
}

# Run the API server with log streaming
/usr/bin/python3 apps/api/server/simple_server.py 2>&1 | while IFS= read -r line; do
    send_log "INFO" "$line"
    echo "$line"
done
