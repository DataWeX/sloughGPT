-- SloughGPT PostgreSQL initialization script
-- Create database schema and initial data

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create custom types
CREATE TYPE user_role AS ENUM ('user', 'admin', 'moderator');
CREATE TYPE conversation_status AS ENUM ('active', 'archived', 'deleted');
CREATE TYPE message_type AS ENUM ('user', 'assistant', 'system');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_api_logs_user_id ON api_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_api_logs_created_at ON api_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_entries(key);
CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache_entries(expires_at);

-- Create full-text search indexes
CREATE INDEX IF NOT EXISTS idx_messages_content_fts ON messages USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_conversations_title_fts ON conversations USING gin(to_tsvector('english', title));

-- Create optimized configuration for JSON operations
ALTER TABLE conversations ALTER COLUMN metadata SET STORAGE EXTERNAL;
ALTER TABLE messages ALTER COLUMN metadata SET STORAGE EXTERNAL;

-- Set up row-level security policies (optional, for multi-tenant setups)
-- ALTER TABLE users ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Create statistics for better query planning
ANALYZE;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sloughgpt;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sloughgpt;

-- Insert default admin user (password: admin123 - change in production)
INSERT INTO users (id, email, username, password_hash, role, is_active, created_at, updated_at)
VALUES (
    uuid_generate_v4(),
    'admin@sloughgpt.local',
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkO3Hj2v/2b2ZdFqBgq8L6kGjqpsUaGC', -- password: admin123
    'admin',
    true,
    NOW(),
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- Create configuration table for system settings
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(255) UNIQUE NOT NULL,
    value TEXT,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default configuration
INSERT INTO system_config (key, value, description) VALUES
    ('max_conversation_length', '100', 'Maximum number of messages in a conversation'),
    ('max_message_length', '4000', 'Maximum length of a single message'),
    ('rate_limit_per_hour', '100', 'Maximum API requests per hour per user'),
    ('file_upload_max_size', '10485760', 'Maximum file upload size in bytes'),
    ('session_timeout', '3600', 'Session timeout in seconds')
ON CONFLICT (key) DO NOTHING;

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for audit log queries
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);

-- Set up automatic cleanup of old logs
CREATE OR REPLACE FUNCTION cleanup_old_logs()
RETURNS void AS $$
BEGIN
    -- Delete API logs older than 30 days
    DELETE FROM api_logs WHERE created_at < NOW() - INTERVAL '30 days';
    
    -- Delete audit logs older than 90 days
    DELETE FROM audit_log WHERE created_at < NOW() - INTERVAL '90 days';
    
    -- Delete expired cache entries
    DELETE FROM cache_entries WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Create scheduled job for cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-old-logs', '0 2 * * *', 'SELECT cleanup_old_logs();');

COMMENT ON TABLE users IS 'User accounts and authentication data';
COMMENT ON TABLE conversations IS 'User conversation sessions';
COMMENT ON TABLE messages IS 'Chat messages in conversations';
COMMENT ON TABLE api_logs IS 'API request and response logging';
COMMENT ON TABLE cache_entries IS 'Application cache storage';
COMMENT ON TABLE system_config IS 'System configuration settings';
COMMENT ON TABLE audit_log IS 'Audit trail for data changes';