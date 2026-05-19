// ============================================================
// documentStore.js —— 文档登记表
//
// pgvector 只适合存 chunk + embedding。
// documents 表负责保存“一个上传文档”的业务信息：
// 原始文件名、服务器保存名、状态、chunk 数量、创建时间等。
// ============================================================

import { query } from './postgres.js';

let ensureTablePromise = null;

function mapDocumentRow(row) {
  if (!row) return null;

  return {
    id: row.id,
    originalName: row.original_name,
    storedFilename: row.stored_filename,
    fileType: row.file_type,
    sourcePath: row.source_path,
    chunkCount: Number(row.chunk_count || 0),
    status: row.status,
    errorMessage: row.error_message,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

export async function ensureDocumentTable() {
  if (!ensureTablePromise) {
    ensureTablePromise = query(`
      CREATE EXTENSION IF NOT EXISTS pgcrypto;

      CREATE TABLE IF NOT EXISTS documents (
        id uuid NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
        original_name text NOT NULL,
        stored_filename text NOT NULL UNIQUE,
        file_type text NOT NULL,
        source_path text NOT NULL,
        chunk_count integer NOT NULL DEFAULT 0,
        status text NOT NULL DEFAULT 'indexing',
        error_message text,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now()
      );

      CREATE INDEX IF NOT EXISTS idx_documents_stored_filename
        ON documents(stored_filename);

      CREATE INDEX IF NOT EXISTS idx_documents_status
        ON documents(status);
    `).catch((error) => {
      ensureTablePromise = null;
      throw error;
    });
  }

  await ensureTablePromise;
}

export async function createDocumentRecord({
  originalName,
  storedFilename,
  fileType,
  sourcePath,
}) {
  await ensureDocumentTable();

  const result = await query(
    `
      INSERT INTO documents (
        original_name,
        stored_filename,
        file_type,
        source_path,
        status
      )
      VALUES ($1, $2, $3, $4, 'indexing')
      RETURNING *
    `,
    [originalName, storedFilename, fileType, sourcePath]
  );

  return mapDocumentRow(result.rows[0]);
}

export async function markDocumentIndexed(documentId, chunkCount) {
  await ensureDocumentTable();

  const result = await query(
    `
      UPDATE documents
      SET chunk_count = $2,
          status = 'indexed',
          error_message = NULL,
          updated_at = now()
      WHERE id = $1
      RETURNING *
    `,
    [documentId, chunkCount]
  );

  return mapDocumentRow(result.rows[0]);
}

export async function markDocumentFailed(documentId, errorMessage) {
  await ensureDocumentTable();

  const result = await query(
    `
      UPDATE documents
      SET status = 'failed',
          error_message = $2,
          updated_at = now()
      WHERE id = $1
      RETURNING *
    `,
    [documentId, errorMessage]
  );

  return mapDocumentRow(result.rows[0]);
}

export async function listDocuments() {
  await ensureDocumentTable();

  const result = await query(`
    SELECT *
    FROM documents
    ORDER BY created_at DESC
  `);

  return result.rows.map(mapDocumentRow);
}

export async function listIndexedDocuments() {
  await ensureDocumentTable();

  const result = await query(`
    SELECT *
    FROM documents
    WHERE status = 'indexed'
    ORDER BY created_at DESC
  `);

  return result.rows.map(mapDocumentRow);
}

export async function getDocumentByStoredFilename(storedFilename) {
  await ensureDocumentTable();

  const result = await query(
    `
      SELECT *
      FROM documents
      WHERE stored_filename = $1
      LIMIT 1
    `,
    [storedFilename]
  );

  return mapDocumentRow(result.rows[0]);
}

export async function findIndexedDocumentByIdentifier(identifier) {
  await ensureDocumentTable();
  const value = identifier?.trim();
  if (!value) return null;

  // 支持三种常见输入：
  // 1. documentId，例如 306da16c-...
  // 2. 原始文件名，例如 tang.txt
  // 3. 服务器保存名，例如 1779084707715.txt
  const result = await query(
    `
      SELECT *
      FROM documents
      WHERE status = 'indexed'
        AND (
          id::text = $1
          OR original_name = $1
          OR stored_filename = $1
        )
      ORDER BY created_at DESC
      LIMIT 1
    `,
    [value]
  );

  return mapDocumentRow(result.rows[0]);
}

export async function deleteDocumentRecordByStoredFilename(storedFilename) {
  await ensureDocumentTable();

  const result = await query(
    `
      DELETE FROM documents
      WHERE stored_filename = $1
      RETURNING *
    `,
    [storedFilename]
  );

  return mapDocumentRow(result.rows[0]);
}

export async function deleteDocumentRecordsNotInStoredFilenames(validStoredFilenames) {
  await ensureDocumentTable();

  if (!validStoredFilenames.length) {
    const result = await query('DELETE FROM documents RETURNING stored_filename');
    return result.rowCount || 0;
  }

  const result = await query(
    `
      DELETE FROM documents
      WHERE NOT (stored_filename = ANY($1::text[]))
      RETURNING stored_filename
    `,
    [validStoredFilenames]
  );

  return result.rowCount || 0;
}

export async function upsertDocumentRecordForExistingFile({
  originalName,
  storedFilename,
  fileType,
  sourcePath,
  chunkCount,
}) {
  await ensureDocumentTable();

  const status = chunkCount > 0 ? 'indexed' : 'uploaded';
  const result = await query(
    `
      INSERT INTO documents (
        original_name,
        stored_filename,
        file_type,
        source_path,
        chunk_count,
        status
      )
      VALUES ($1, $2, $3, $4, $5, $6)
      ON CONFLICT (stored_filename)
      DO UPDATE SET
        file_type = EXCLUDED.file_type,
        source_path = EXCLUDED.source_path,
        chunk_count = EXCLUDED.chunk_count,
        status = EXCLUDED.status,
        updated_at = now()
      RETURNING *
    `,
    [originalName, storedFilename, fileType, sourcePath, chunkCount, status]
  );

  return mapDocumentRow(result.rows[0]);
}
