// ============================================================
// server.js —— Express Web 服务器入口
//
// 职责：
//  1. 提供前端页面
//  2. 接收文件上传（图片存路径，文档走 RAG）
//  3. 接收聊天消息，直接转发给 Agent
//
// 图片场景流程（Agent 负责调工具）：
//  前端上传图片 → 保存到 uploads/ → 把路径告诉 Agent
//  → Agent 调用 image_analysis 工具识别食材
//  → Agent 调用 web_search 工具搜索菜谱
//  → Agent 整理输出报告
// ============================================================

import 'dotenv/config';

import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import Database from 'better-sqlite3';
import { invokeAgent, streamAgent } from './src/agent.js';
import {
  attachDocumentIdToIndex,
  cleanupMissingDocumentIndexes,
  deleteDocumentIndexById,
  deleteDocumentIndex,
  getDocumentIndexStats,
  processDocument,
} from './src/rag.js';
import {
  createDocumentRecord,
  deleteDocumentRecordByStoredFilename,
  deleteDocumentRecordsNotInStoredFilenames,
  ensureDocumentTable,
  getDocumentByStoredFilename,
  listDocuments,
  markDocumentFailed,
  markDocumentIndexed,
  upsertDocumentRecordForExistingFile,
} from './src/documentStore.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;
const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp']);
const DOCUMENT_EXTS = new Set(['.pdf', '.txt']);

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
// 静态服务 uploads 目录，让前端能通过 /uploads/xxx.jpg 访问已上传的图片
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// -------------------------------------------------------
// Multer 文件上传配置
// -------------------------------------------------------
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${Date.now()}${ext}`);
  },
});

const fileFilter = (req, file, cb) => {
  const ext = path.extname(file.originalname).toLowerCase();
  DOCUMENT_EXTS.has(ext) || IMAGE_EXTS.has(ext)
    ? cb(null, true)
    : cb(new Error(`不支持的格式：${ext}`), false);
};

const upload = multer({ storage, fileFilter });

function isDocumentFilename(filename) {
  return DOCUMENT_EXTS.has(path.extname(filename).toLowerCase());
}

function isImageFilename(filename) {
  return IMAGE_EXTS.has(path.extname(filename).toLowerCase());
}

function getSafeUploadFilename(filename) {
  const safeFilename = path.basename(filename || '');

  if (!safeFilename || safeFilename !== filename) {
    throw new Error('非法文件名');
  }

  return safeFilename;
}

// -------------------------------------------------------
// 路由：POST /upload —— 处理文件上传
// -------------------------------------------------------
app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    const file = req.file;
    if (!file) return res.status(400).json({ error: '没有收到文件' });

    const ext = path.extname(file.originalname).toLowerCase();
    const isImage = IMAGE_EXTS.has(ext);
    const isDocument = DOCUMENT_EXTS.has(ext);
    const storedFilename = path.basename(file.path);

    if (isDocument) {
      console.log(`[Server] 收到文档: ${file.originalname}`);

      // 先创建 documents 业务记录，拿到稳定 documentId。
      // 后续每个 chunk 的 metadata 都会带上这个 documentId。
      const documentRecord = await createDocumentRecord({
        originalName: file.originalname,
        storedFilename,
        fileType: ext.replace('.', ''),
        sourcePath: `uploads/${storedFilename}`,
      });

      let indexedDocument = documentRecord;
      let chunkCount = 0;
      try {
        chunkCount = await processDocument(file.path, {
          originalName: file.originalname,
          documentId: documentRecord.id,
        });
        indexedDocument = await markDocumentIndexed(documentRecord.id, chunkCount);
      } catch (indexError) {
        await markDocumentFailed(documentRecord.id, indexError.message);
        throw indexError;
      }

      res.json({
        success: true,
        type: 'document',
        message: `文档上传成功，已切分为 ${chunkCount} 个块并建立索引`,
        documentId: indexedDocument.id,
        filePath: file.path,
        filename: storedFilename,
        originalName: file.originalname,
        chunkCount,
      });
    } else if (isImage) {
      console.log(`[Server] 收到图片: ${file.originalname}`);
      res.json({
        success: true,
        type: 'image',
        message: '图片上传成功，发送消息时我会自动分析食材',
        filePath: file.path,
        filename: storedFilename,
        originalName: file.originalname,
      });
    }
  } catch (err) {
    console.error('[Server] 上传失败:', err);
    res.status(500).json({ error: err.message });
  }
});

// -------------------------------------------------------
// 路由：POST /chat —— 接收消息，转发给 Agent
//
// 服务端不做任何预处理，图片分析和搜索全部由 Agent 调工具完成。
// 服务端只把图片路径附在消息里，让 Agent 知道有图片可以分析。
// -------------------------------------------------------
app.post('/chat', async (req, res) => {
  try {
    const {
      message,
      threadId = 'default',
      imagePath,
      selectedDocumentIds = [],
    } = req.body;

    if (!message) return res.status(400).json({ error: '消息不能为空' });
    const documentScopeIds = Array.isArray(selectedDocumentIds)
      ? selectedDocumentIds.filter((id) => typeof id === 'string' && id.trim())
      : [];

    console.log(`[Server] 收到消息 (thread: ${threadId}): ${message}`);

    const fullMessage = imagePath
      ? `${message}\n\n[用户上传了图片，服务器路径：${imagePath}，请调用 image_analysis 工具分析这张图片中的食材]`
      : message;

    // SSE 流式响应
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // streamAgent 现在返回结构化事件：
    // - text：模型最终回答文本
    // - tool_start：Agent 决定调用工具
    // - tool_result：工具执行完成或失败
    for await (const event of streamAgent(fullMessage, threadId, {
      hasImage: Boolean(imagePath),
      selectedDocumentIds: documentScopeIds,
    })) {
      const payload = typeof event === 'string'
        ? { type: 'text', content: event }
        : event;
      res.write(`data: ${JSON.stringify(payload)}\n\n`);
    }

    // 发送结束标记
    res.write(`data: [DONE]\n\n`);
    res.end();
  } catch (err) {
    console.error('[Server] 处理失败:', err);
    // 如果还没开始写 SSE 头，返回 JSON 错误
    if (!res.headersSent) {
      res.status(500).json({ error: err.message });
    } else {
      res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    }
  }
});

// -------------------------------------------------------
// 路由：GET /threads —— 获取所有历史会话列表
//
// 从 checkpoint.db 的 writes 表中读取每个 thread 的第一条
// 用户消息作为预览，按时间倒序排列。
// -------------------------------------------------------
app.get('/threads', (req, res) => {
  try {
    const db = new Database('./checkpoint.db', { readonly: true });
    // 获取所有 thread_id，以及每个 thread 的第一条用户消息作为预览
    const rows = db.prepare(`
      SELECT w.thread_id, w.value
      FROM writes w
      WHERE w.channel = 'messages'
        AND w.checkpoint_ns = ''
      GROUP BY w.thread_id
      HAVING w.rowid = MIN(w.rowid)
      ORDER BY w.rowid DESC
    `).all();
    db.close();

    const threads = rows.map((row) => {
      let preview = '新对话';
      try {
        const msgs = JSON.parse(row.value);
        if (msgs.length > 0) {
          const content = msgs[0]?.kwargs?.content || '';
          // 取前 50 个字符作为预览，去掉图片路径提示
          preview = content.replace(/\n\n\[用户上传了图片[\s\S]*?\]/, '').trim();
          if (preview.length > 50) preview = preview.slice(0, 50) + '...';
          if (!preview) preview = '图片分析';
        }
      } catch {}
      return { threadId: row.thread_id, preview };
    });

    res.json({ threads });
  } catch (err) {
    console.error('[Server] 获取会话列表失败:', err);
    res.json({ threads: [] });
  }
});

// -------------------------------------------------------
// 路由：GET /threads/:id/messages —— 获取指定会话的历史消息
//
// 从 writes 表中读取所有消息，解析为前端可用的格式。
// 只返回 HumanMessage 和 AIMessage，跳过 ToolMessage。
// -------------------------------------------------------
app.get('/threads/:id/messages', (req, res) => {
  try {
    const db = new Database('./checkpoint.db', { readonly: true });
    const rows = db.prepare(`
      SELECT value FROM writes
      WHERE thread_id = ? AND channel = 'messages' AND checkpoint_ns = ''
      ORDER BY rowid ASC
    `).all(req.params.id);
    db.close();

    const messages = [];
    for (const row of rows) {
      try {
        const parsed = JSON.parse(row.value);
        if (!Array.isArray(parsed) || parsed.length === 0) continue;
        const msg = parsed[0];
        const type = msg?.id?.[msg.id.length - 1]; // "HumanMessage" or "AIMessage"
        const content = msg?.kwargs?.content || '';
        if (!content || typeof content !== 'string') continue;

        if (type === 'HumanMessage') {
          // 提取图片路径（如果有）
          const imgMatch = content.match(/服务器路径：(uploads\/[^\s,，\]]+)/);
          const cleanContent = content.replace(/\n\n\[用户上传了图片[\s\S]*?\]/, '').trim();
          messages.push({
            role: 'user',
            content: cleanContent,
            imagePath: imgMatch ? imgMatch[1] : null,
          });
        } else if (type === 'AIMessage') {
          messages.push({ role: 'assistant', content });
        }
      } catch {}
    }

    res.json({ messages });
  } catch (err) {
    console.error('[Server] 获取消息失败:', err);
    res.json({ messages: [] });
  }
});

// -------------------------------------------------------
// 路由：POST /threads/batch-delete —— 批量删除会话
// -------------------------------------------------------
app.post('/threads/batch-delete', (req, res) => {
  try {
    const { threadIds } = req.body;
    if (!Array.isArray(threadIds) || threadIds.length === 0) {
      return res.status(400).json({ error: '没有选择要删除的会话' });
    }
    const db = new Database('./checkpoint.db');
    const placeholders = threadIds.map(() => '?').join(',');
    db.prepare(`DELETE FROM writes WHERE thread_id IN (${placeholders})`).run(...threadIds);
    db.prepare(`DELETE FROM checkpoints WHERE thread_id IN (${placeholders})`).run(...threadIds);
    db.close();
    console.log(`[Server] 批量删除 ${threadIds.length} 个会话`);
    res.json({ success: true, deleted: threadIds.length });
  } catch (err) {
    console.error('[Server] 批量删除会话失败:', err);
    res.status(500).json({ error: err.message });
  }
});

// -------------------------------------------------------
// 路由：POST /files/batch-delete —— 批量删除文件
// -------------------------------------------------------
app.post('/files/batch-delete', async (req, res) => {
  try {
    const { filenames } = req.body;
    if (!Array.isArray(filenames) || filenames.length === 0) {
      return res.status(400).json({ error: '没有选择要删除的文件' });
    }

    let deleted = 0;
    let deletedChunks = 0;

    for (const filename of filenames) {
      const safeFilename = getSafeUploadFilename(filename);
      const filePath = path.join(__dirname, 'uploads', safeFilename);

      // 文档文件删除时，同步删除 pgvector 中对应的 chunk 向量。
      // 图片没有入向量库，所以不需要删索引。
      if (isDocumentFilename(safeFilename)) {
        const documentRecord = await getDocumentByStoredFilename(safeFilename);
        if (documentRecord?.id) {
          deletedChunks += await deleteDocumentIndexById(documentRecord.id);
        }
        // 兼容旧数据：早期 chunk metadata 里没有 documentId，只能用 storedFilename 删除。
        deletedChunks += await deleteDocumentIndex(safeFilename);
        await deleteDocumentRecordByStoredFilename(safeFilename);
      }

      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        deleted++;
      }
    }

    console.log(`[Server] 批量删除 ${deleted} 个文件，清理 ${deletedChunks} 条文档向量`);
    res.json({ success: true, deleted, deletedChunks });
  } catch (err) {
    console.error('[Server] 批量删除文件失败:', err);
    res.status(500).json({ error: err.message });
  }
});

// -------------------------------------------------------
// 路由：DELETE /files/:filename —— 删除单个上传文件
// -------------------------------------------------------
app.delete('/files/:filename', async (req, res) => {
  try {
    const filename = getSafeUploadFilename(req.params.filename);
    const filePath = path.join(__dirname, 'uploads', filename);
    let deleted = 0;
    let deletedChunks = 0;

    // 先删向量索引，再删物理文件。
    // 即使物理文件已经不存在，也要尽量清理 pgvector 里的旧索引。
    if (isDocumentFilename(filename)) {
      const documentRecord = await getDocumentByStoredFilename(filename);
      if (documentRecord?.id) {
        deletedChunks += await deleteDocumentIndexById(documentRecord.id);
      }
      // 兼容旧数据：早期 chunk metadata 里没有 documentId，只能用 storedFilename 删除。
      deletedChunks += await deleteDocumentIndex(filename);
      await deleteDocumentRecordByStoredFilename(filename);
    }

    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      deleted = 1;
    }

    console.log(`[Server] 删除文件 ${filename}，文件删除=${deleted}，向量删除=${deletedChunks}`);
    res.json({ success: true, deleted, deletedChunks });
  } catch (err) {
    console.error('[Server] 删除文件失败:', err);
    res.status(500).json({ error: err.message });
  }
});

// -------------------------------------------------------
// 路由：GET /files —— 获取已上传的文件列表
// -------------------------------------------------------
app.get('/files', async (req, res) => {
  try {
    const uploadsDir = path.join(__dirname, 'uploads');
    const uploadFilenames = fs.existsSync(uploadsDir)
      ? fs.readdirSync(uploadsDir)
      : [];

    // 文档列表从 documents 表读取，保留 originalName、documentId、chunkCount 等业务信息。
    const documents = await listDocuments();
    const documentFiles = documents.map((doc) => ({
      documentId: doc.id,
      filename: doc.storedFilename,
      originalName: doc.originalName,
      type: 'document',
      path: doc.sourcePath,
      status: doc.status,
      chunkCount: doc.chunkCount,
      errorMessage: doc.errorMessage,
      time: doc.createdAt ? new Date(doc.createdAt).getTime() : 0,
    }));

    // 图片不进入 RAG 知识库，仍然从 uploads/ 目录扫描。
    const imageFiles = uploadFilenames
      .filter(isImageFilename)
      .map((filename) => {
        const stat = fs.statSync(path.join(uploadsDir, filename));
        return {
          filename,
          originalName: filename,
          type: 'image',
          path: `uploads/${filename}`,
          time: stat.mtimeMs,
        };
      });

    const files = [...documentFiles, ...imageFiles];
    // 按上传时间倒序
    files.sort((a, b) => b.time - a.time);
    res.json({ files });
  } catch (err) {
    console.error('[Server] 获取文件列表失败:', err);
    res.json({ files: [] });
  }
});

// -------------------------------------------------------
// 启动服务
//
// 现在向量库已经换成 pgvector，索引会持久化在 PostgreSQL 中。
// 因此启动时不能再扫描 uploads/ 重建索引，否则每次重启都会重复写入 chunk。
// 启动时只做轻量同步：
//  1. 确保 documents 表存在
//  2. 把旧版本已有的 uploads 文档补进 documents 表
//  3. 清理 uploads/ 里不存在的 documents 记录和 pgvector 孤儿索引
// -------------------------------------------------------
async function syncDocumentRegistry() {
  try {
    await ensureDocumentTable();

    const uploadsDir = path.join(__dirname, 'uploads');
    const validDocumentFilenames = fs.existsSync(uploadsDir)
      ? fs.readdirSync(uploadsDir).filter(isDocumentFilename)
      : [];

    for (const storedFilename of validDocumentFilenames) {
      const fileType = path.extname(storedFilename).replace('.', '').toLowerCase();
      const sourcePath = `uploads/${storedFilename}`;
      const stats = await getDocumentIndexStats(storedFilename);
      const originalName = stats.metadata?.filename || storedFilename;

      const documentRecord = await upsertDocumentRecordForExistingFile({
        originalName,
        storedFilename,
        fileType,
        sourcePath,
        chunkCount: stats.chunkCount,
      });

      await attachDocumentIdToIndex(storedFilename, documentRecord.id);
    }

    const deletedDocuments = await deleteDocumentRecordsNotInStoredFilenames(validDocumentFilenames);
    const deletedChunks = await cleanupMissingDocumentIndexes(validDocumentFilenames);
    if (deletedDocuments > 0 || deletedChunks > 0) {
      console.log(
        `[启动] 文档登记同步完成：清理 ${deletedDocuments} 条文档记录，${deletedChunks} 条孤儿向量索引`
      );
    }
  } catch (err) {
    // 数据库没有启动时，不阻塞普通聊天和前端页面启动。
    // 用户上传文档或检索文档时，接口会再返回明确错误。
    console.warn(`[启动] 文档登记同步跳过: ${err.message}`);
  }
}

syncDocumentRegistry().finally(() => {
  app.listen(PORT, () => {
    console.log(`\n🚀 私厨问答管家已启动`);
    console.log(`📡 访问地址：http://localhost:${PORT}`);
    console.log(`📁 上传目录：./uploads/`);
    console.log(`💾 对话存储：./checkpoint.db\n`);
  });
});
