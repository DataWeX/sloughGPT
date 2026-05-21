/** Shell snippets for the API docs quick-examples panel (no shell interpolation — safe for display). */
export function apiDocsCurlExamplesBlock(baseUrl: string): string {
  return `# Health check
curl ${baseUrl}/health

# Generate text
curl -X POST ${baseUrl}/inference/generate \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello world", "max_new_tokens": 50}'

# Streaming (SSE)
curl -X POST ${baseUrl}/inference/generate/stream \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello", "max_new_tokens": 100}'

# Run benchmark
curl -X POST "${baseUrl}/benchmark/run?max_new_tokens=20" \\
  -H "Content-Type: application/json" -d '{}'

# Create experiment
curl -X POST "${baseUrl}/experiments?name=test&description=Testing"

# Export model
curl -X POST ${baseUrl}/model/export \\
  -H "Content-Type: application/json" \\
  -d '{"output_path": "models/exported", "format": "torch"}'`
}
