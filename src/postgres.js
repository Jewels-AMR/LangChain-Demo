// ============================================================
// postgres.js —— PostgreSQL 连接池
//
// 普通业务表（documents）和 pgvector 向量表都会连接同一个数据库。
// 把连接配置放在这里，可以避免每个模块重复读取环境变量。
// ============================================================

import pg from 'pg';

const { Pool } = pg;

// 这里不写 type: 'postgres'。
// type 是 LangChain PGVectorStore 示例里的字段，pg.Pool 本身不需要它。
export const postgresConnectionOptions = {
  host: process.env.POSTGRES_HOST || '127.0.0.1',
  port: Number(process.env.POSTGRES_PORT || 5432),
  database: process.env.POSTGRES_DB || 'agent_demo',
  user: process.env.POSTGRES_USER || 'agent',
  password: process.env.POSTGRES_PASSWORD || 'agent_password',
};

export const postgresPool = new Pool(postgresConnectionOptions);

export async function query(text, params = []) {
  return postgresPool.query(text, params);
}
