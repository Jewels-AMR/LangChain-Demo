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
import { processDocument } from './src/rag.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;

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
  const allowed = ['.pdf', '.txt', '.jpg', '.jpeg', '.png', '.webp'];
  const ext = path.extname(file.originalname).toLowerCase();
  allowed.includes(ext) ? cb(null, true) : cb(new Error(`不支持的格式：${ext}`), false);
};

const upload = multer({ storage, fileFilter });

// -------------------------------------------------------
// 路由：POST /upload —— 处理文件上传
// -------------------------------------------------------
app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    const file = req.file;
    if (!file) return res.status(400).json({ error: '没有收到文件' });

    const ext = path.extname(file.originalname).toLowerCase();
    const isImage = ['.jpg', '.jpeg', '.png', '.webp'].includes(ext);
    const isDocument = ['.pdf', '.txt'].includes(ext);

    if (isDocument) {
      console.log(`[Server] 收到文档: ${file.originalname}`);
      const chunkCount = await processDocument(file.path, {
        originalName: file.originalname,
      });
      res.json({
        success: true,
        type: 'document',
        message: `文档上传成功，已切分为 ${chunkCount} 个块并建立索引`,
        filePath: file.path,
      });
    } else if (isImage) {
      console.log(`[Server] 收到图片: ${file.originalname}`);
      res.json({
        success: true,
        type: 'image',
        message: '图片上传成功，发送消息时我会自动分析食材',
        filePath: file.path,
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
    const { message, threadId = 'default', imagePath } = req.body;

    if (!message) return res.status(400).json({ error: '消息不能为空' });

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
app.post('/files/batch-delete', (req, res) => {
  try {
    const { filenames } = req.body;
    if (!Array.isArray(filenames) || filenames.length === 0) {
      return res.status(400).json({ error: '没有选择要删除的文件' });
    }
    let deleted = 0;
    for (const filename of filenames) {
      const filePath = path.join(__dirname, 'uploads', filename);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        deleted++;
      }
    }
    console.log(`[Server] 批量删除 ${deleted} 个文件`);
    res.json({ success: true, deleted });
  } catch (err) {
    console.error('[Server] 批量删除文件失败:', err);
    res.status(500).json({ error: err.message });
  }
});

// -------------------------------------------------------
// 路由：GET /files —— 获取已上传的文件列表
// -------------------------------------------------------
app.get('/files', (req, res) => {
  try {
    const uploadsDir = path.join(__dirname, 'uploads');
    const files = fs.readdirSync(uploadsDir).map((filename) => {
      const ext = path.extname(filename).toLowerCase();
      const isImage = ['.jpg', '.jpeg', '.png', '.webp'].includes(ext);
      const stat = fs.statSync(path.join(uploadsDir, filename));
      return {
        filename,
        type: isImage ? 'image' : 'document',
        path: `uploads/${filename}`,
        time: stat.mtimeMs,
      };
    });
    // 按上传时间倒序
    files.sort((a, b) => b.time - a.time);
    res.json({ files });
  } catch (err) {
    res.json({ files: [] });
  }
});

// -------------------------------------------------------
// 启动时自动重建向量索引
//
// MemoryVectorStore 是内存存储，重启后向量丢失。
// 扫描 uploads/ 里已有的文档文件，自动重新向量化，
// 这样重启后不用重新上传文档。
// -------------------------------------------------------
async function rebuildIndex() {
  const uploadsDir = path.join(__dirname, 'uploads');
  if (!fs.existsSync(uploadsDir)) return;

  const docExts = ['.pdf', '.txt'];
  const docFiles = fs.readdirSync(uploadsDir).filter((f) =>
    docExts.includes(path.extname(f).toLowerCase())
  );

  if (docFiles.length === 0) return;

  console.log(`[启动] 发现 ${docFiles.length} 个文档，正在重建向量索引...`);

  for (const filename of docFiles) {
    try {
      const filePath = path.join(uploadsDir, filename);
      const chunkCount = await processDocument(filePath);
      console.log(`[启动] ${filename} → ${chunkCount} 个块`);
    } catch (err) {
      console.error(`[启动] ${filename} 索引失败:`, err.message);
    }
  }

  console.log(`[启动] 向量索引重建完成`);
}

rebuildIndex().then(() => {
  app.listen(PORT, () => {
    console.log(`\n🚀 私厨问答管家已启动`);
    console.log(`📡 访问地址：http://localhost:${PORT}`);
    console.log(`📁 上传目录：./uploads/`);
    console.log(`💾 对话存储：./checkpoint.db\n`);
  });
});
