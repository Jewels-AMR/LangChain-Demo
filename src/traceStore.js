// ============================================================
// traceStore.js —— Agent loop 前端调试轨迹持久化
//
// LangGraph checkpoint 负责“模型记忆”。
// 这里的 agent_turn_traces 只负责保存 UI 要回放的工具调用过程：
// tool_start / tool_result / rag_debug / sources。
// ============================================================

import Database from 'better-sqlite3';

const DB_PATH = './checkpoint.db';
let tableReady = false;

function parseJson(value, fallback) {
  try {
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function stringifyJson(value, fallback) {
  try {
    return JSON.stringify(value ?? fallback);
  } catch {
    return JSON.stringify(fallback);
  }
}

function getTraceSources(events) {
  const reversedEvents = [...events].reverse();
  const sourceEvent = reversedEvents.find((event) =>
    event?.type === 'sources' && Array.isArray(event.sources)
  );
  if (sourceEvent) return sourceEvent.sources;

  const ragEvent = reversedEvents.find((event) =>
    event?.type === 'rag_debug' && Array.isArray(event.sources)
  );
  return ragEvent?.sources || [];
}

function mapTraceRow(row) {
  const events = parseJson(row.events, []);
  const ragDebugEvents = events.filter((event) => event?.type === 'rag_debug');

  return {
    id: row.id,
    turnId: row.turn_id,
    threadId: row.thread_id,
    userMessage: row.user_message,
    assistantMessage: row.assistant_message,
    imagePath: row.image_path,
    selectedDocumentIds: parseJson(row.selected_document_ids, []),
    events,
    ragDebugEvents,
    sources: getTraceSources(events),
    createdAt: row.created_at,
  };
}

export function ensureTraceTable() {
  if (tableReady) return;

  const db = new Database(DB_PATH);
  try {
    db.exec(`
      CREATE TABLE IF NOT EXISTS agent_turn_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT NOT NULL,
        turn_id TEXT NOT NULL UNIQUE,
        user_message TEXT NOT NULL,
        assistant_message TEXT NOT NULL DEFAULT '',
        image_path TEXT,
        selected_document_ids TEXT NOT NULL DEFAULT '[]',
        events TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      );

      CREATE INDEX IF NOT EXISTS idx_agent_turn_traces_thread_id
        ON agent_turn_traces(thread_id);

      CREATE INDEX IF NOT EXISTS idx_agent_turn_traces_created_at
        ON agent_turn_traces(created_at);
    `);
    tableReady = true;
  } finally {
    db.close();
  }
}

export function saveAgentTurnTrace({
  threadId,
  turnId,
  userMessage,
  assistantMessage = '',
  imagePath = null,
  selectedDocumentIds = [],
  events = [],
}) {
  ensureTraceTable();

  const db = new Database(DB_PATH);
  try {
    db.prepare(`
      INSERT INTO agent_turn_traces (
        thread_id,
        turn_id,
        user_message,
        assistant_message,
        image_path,
        selected_document_ids,
        events
      )
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(turn_id)
      DO UPDATE SET
        assistant_message = excluded.assistant_message,
        selected_document_ids = excluded.selected_document_ids,
        events = excluded.events
    `).run(
      threadId,
      turnId,
      userMessage,
      assistantMessage,
      imagePath,
      stringifyJson(selectedDocumentIds, []),
      stringifyJson(events, [])
    );
  } finally {
    db.close();
  }
}

export function listAgentTurnTraces(threadId) {
  ensureTraceTable();

  const db = new Database(DB_PATH, { readonly: true });
  try {
    const rows = db.prepare(`
      SELECT *
      FROM agent_turn_traces
      WHERE thread_id = ?
      ORDER BY id ASC
    `).all(threadId);

    return rows.map(mapTraceRow);
  } finally {
    db.close();
  }
}

export function deleteAgentTracesByThread(threadId) {
  ensureTraceTable();

  const db = new Database(DB_PATH);
  try {
    const result = db.prepare(`
      DELETE FROM agent_turn_traces
      WHERE thread_id = ?
    `).run(threadId);

    return result.changes || 0;
  } finally {
    db.close();
  }
}

export function deleteAgentTracesByThreads(threadIds) {
  const ids = Array.isArray(threadIds)
    ? threadIds.filter((threadId) => typeof threadId === 'string' && threadId)
    : [];
  if (ids.length === 0) return 0;

  ensureTraceTable();

  const db = new Database(DB_PATH);
  try {
    const placeholders = ids.map(() => '?').join(',');
    const result = db.prepare(`
      DELETE FROM agent_turn_traces
      WHERE thread_id IN (${placeholders})
    `).run(...ids);

    return result.changes || 0;
  } finally {
    db.close();
  }
}
