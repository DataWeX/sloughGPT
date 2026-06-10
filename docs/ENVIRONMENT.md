# SloughGPT Environment Variables

Complete reference for all environment variables used in SloughGPT.

## Quick Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MAN_API_KEY` | Yes | - | API key for authentication |
| `MAN_JWT_SECRET` | Yes | - | Secret for JWT token signing |
| `MAN_ENV` | No | `development` | Environment mode |
| `MAN_HOST` | No | `0.0.0.0` | Server host |
| `MAN_PORT` | No | `8000` | Server port |
| `MAN_RELOAD` | No | `false` | Enable auto-reload on file changes |

**Legacy names:** older docs and images used a typo (`SLAUGHGPT_*`). The server still accepts `SLAUGHGPT_API_KEY`, `SLAUGHGPT_JWT_SECRET`, and `SLAUGHGPT_API_KEYS` if the `MAN_*` counterparts are unset. Prefer `MAN_*` for new deployments.

---

## Authentication

### MAN_API_KEY
**Required in production**

API key for authenticating requests.

```bash
# Generate a secure key
openssl rand -hex 32

# Set in .env
MAN_API_KEY=your-generated-key-here
```

### MAN_API_KEYS
**Optional**

Comma-separated list of multiple valid API keys.

```bash
MAN_API_KEYS=key1,key2,key3
```

### MAN_JWT_SECRET
**Required in production**

Secret key for signing JWT tokens.

```bash
# Generate a secure secret
openssl rand -hex 64

# Set in .env
MAN_JWT_SECRET=your-64-character-secret
```

---

## Server Configuration

### MAN_ENV
**Optional**

Environment mode.

```bash
MAN_ENV=development  # or production
```

### MAN_HOST
**Optional**

Server bind address.

```bash
MAN_HOST=0.0.0.0  # Default
```

### MAN_PORT
**Optional**

Server port.

```bash
MAN_PORT=8000  # Default
```

### MAN_RELOAD
**Optional**

Enable uvicorn auto-reload on Python file changes.

```bash
MAN_RELOAD=true  # Default: false
```

---

## Legacy / Deprecated

The following environment variable names are accepted as fallbacks:

| Legacy Name | Modern Name |
|-------------|-------------|
| `SLAUGHGPT_API_KEY` | `MAN_API_KEY` |
| `SLAUGHGPT_JWT_SECRET` | `MAN_JWT_SECRET` |
| `SLAUGHGPT_API_KEYS` | `MAN_API_KEYS` |

These are read from `settings.py` if the `MAN_*` variant is unset. New deployments should use only `MAN_*` names.
