-- 1. 创建数据库

SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = 'zrkb'
  AND pid <> pg_backend_pid();
DROP DATABASE zrkb;
CREATE DATABASE zrkb;

-- 2. 创建向量扩展，支持向量数据库

CREATE EXTENSION vector;
SELECT * FROM pg_extension WHERE extname = 'vector';

-- 3. 创建表格

DROP TABLE IF EXISTS max_kb_file CASCADE;
DROP TABLE IF EXISTS max_kb_dataset CASCADE;
DROP TABLE IF EXISTS max_kb_document CASCADE;
DROP TABLE IF EXISTS max_kb_paragraph CASCADE;
DROP TABLE IF EXISTS max_kb_embedding CASCADE;

CREATE TABLE max_kb_file (
  id BIGINT PRIMARY KEY,
  md5 VARCHAR(32) NOT NULL,
  filename VARCHAR(256) NOT NULL,
  file_size BIGINT NOT NULL,
  user_id VARCHAR(32),
  platform VARCHAR(256) NOT NULL,
  region_name VARCHAR(32),
  bucket_name VARCHAR(256) NOT NULL,
  file_id VARCHAR(256) NOT NULL,
  target_name VARCHAR(256) NOT NULL,
  tags JSON,
  creator VARCHAR(256) DEFAULT '',
  create_time TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updater VARCHAR(256) DEFAULT '',
  update_time TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted SMALLINT NOT NULL DEFAULT 0,
  tenant_id BIGINT NOT NULL DEFAULT 0
);



CREATE TABLE max_kb_dataset (
  id BIGINT PRIMARY KEY,
  name VARCHAR NOT NULL,
  description VARCHAR,
  type VARCHAR,
  meta JSONB,
  user_id VARCHAR NOT NULL,
  remark VARCHAR(256),
  creator VARCHAR(256) DEFAULT '',
  create_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updater VARCHAR(256) DEFAULT '',
  update_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted SMALLINT DEFAULT 0,
  tenant_id BIGINT NOT NULL DEFAULT 0
);


CREATE TABLE max_kb_document (
  id BIGINT NOT NULL,
  name VARCHAR NOT NULL,
  char_length INT NOT NULL,
  status VARCHAR NOT NULL,
  is_active BOOLEAN NOT NULL,
  type VARCHAR NOT NULL,
  meta JSONB NOT NULL,
  dataset_id BIGINT NOT NULL,
  hit_handling_method VARCHAR NOT NULL,
  directly_return_similarity FLOAT8 NOT NULL,
  files JSON,
  creator VARCHAR(256) DEFAULT '',
  create_time TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updater VARCHAR(256) DEFAULT '',
  update_time TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted SMALLINT NOT NULL DEFAULT 0,
  tenant_id BIGINT NOT NULL DEFAULT 0
);


CREATE TABLE max_kb_paragraph (
  id BIGINT NOT NULL,
  content VARCHAR NOT NULL,
  title VARCHAR NOT NULL,
  status VARCHAR NOT NULL,
  hit_num INT NOT NULL,
  is_active BOOLEAN NOT NULL,
  dataset_id BIGINT NOT NULL,
  document_id BIGINT NOT NULL,
  creator VARCHAR(256) DEFAULT '',
  create_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updater VARCHAR(256) DEFAULT '',
  update_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted SMALLINT DEFAULT 0,
  tenant_id BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE max_kb_embedding (
  id BIGINT PRIMARY KEY,
  source_id BIGINT NOT NULL,
  source_type VARCHAR NOT NULL,
  is_active BOOLEAN NOT NULL,
  embedding VECTOR(2048) NOT NULL,
  meta JSONB NOT NULL,
  dataset_id BIGINT NOT NULL,
  document_id BIGINT NOT NULL,
  paragraph_id BIGINT NOT NULL,
  search_vector TSVECTOR NOT NULL,
  creator VARCHAR(256) DEFAULT '',
  create_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updater VARCHAR(256) DEFAULT '',
  update_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted SMALLINT DEFAULT 0,
  tenant_id BIGINT NOT NULL DEFAULT 0
);

-- 4. 检查在zrkb数据库中是否有表格

SELECT tablename FROM pg_tables WHERE schemaname = 'public';

-- 查看文件
SELECT * FROM max_kb_embedding;
SELECT * FROM max_kb_file;