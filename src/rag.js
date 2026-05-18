// ============================================================
// rag.js —— RAG 文档处理管道
//
// RAG 分两个阶段：
//
// 【离线阶段】用户上传文件时触发（只做一次）：
//   文件 → 加载 → 切块(chunk) → 向量化(embedding) → 存入向量库
//
// 【在线阶段】用户提问时触发（每次提问都做）：
//   问题 → 向量化 → 在向量库中检索相似块 → 返回相关内容给 Agent
// ============================================================

import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { TextLoader } from '@langchain/classic/document_loaders/fs/text';
import path from 'path';
import {
  addDocumentsToVectorStore,
  deleteDocumentIndexesNotInStoredFilenames,
  deleteDocumentsFromVectorStoreByStoredFilename,
  hasIndexedDocuments,
  searchSimilarDocuments,
} from './vectorStore.js';

// -------------------------------------------------------
// 向量库说明
//
// 具体向量库实现已经下沉到 vectorStore.js。
// rag.js 只负责 RAG 文档处理流程：
//   加载 → 切块 → 补 metadata → 调用向量库适配层
//
// 这样后续把 pgvector 换成 Qdrant/Redis 时，
// 不需要改这里的文档解析和切块逻辑。
// -------------------------------------------------------

/**
 * 根据文件扩展名选择对应的文档加载器
 *
 * 不同格式的文件需要不同的解析方式：
 *  - PDF 有页面结构，需要专门的 PDF 解析库
 *  - TXT 直接按文本读取即可
 *
 * @param {string} filePath - 文件的绝对路径
 * @returns {BaseDocumentLoader} LangChain 文档加载器实例
 */
function getLoader(filePath) {
  const ext = path.extname(filePath).toLowerCase();

  switch (ext) {
    case '.pdf':
      // PDFLoader 会按页拆分文档，每页是一个 Document 对象
      return new PDFLoader(filePath);
    case '.txt':
      // TextLoader 直接读取纯文本
      return new TextLoader(filePath);
    default:
      throw new Error(`暂不支持 ${ext} 格式，目前支持：.pdf .txt`);
  }
}

/**
 * 处理上传的文件，走完整的 RAG 离线阶段
 *
 * 步骤：
 *  1. 加载文件 → 得到若干 Document 对象（每个 Document 含文本和元数据）
 *  2. 切块 → 把长文档切成小段，避免单块信息过多或过少
 *  3. 向量化 + 存入向量库 → 为后续检索做准备
 *
 * @param {string} filePath - 上传文件在服务器上的路径
 * @param {{ originalName?: string }} options - 上传时保留的文件信息
 * @returns {number} 最终存入向量库的块数量
 */
export async function processDocument(filePath, options = {}) {
  // --- 第一步：加载文件 ---
  const loader = getLoader(filePath);
  const docs = await loader.load();
  console.log(`[RAG] 文件加载完成，共 ${docs.length} 个文档段`);

  const storedFilename = path.basename(filePath);
  const filename = options.originalName || storedFilename;
  const ext = path.extname(filePath).toLowerCase();
  const sourcePath = `uploads/${storedFilename}`;

  // metadata 是每个 chunk 的"身份证"：
  // 后续检索命中时，可以知道内容来自哪个文件、哪一页、哪一个 chunk。
  const docsWithMetadata = docs.map((doc, docIndex) => ({
    ...doc,
    metadata: {
      ...doc.metadata,
      filename,
      storedFilename,
      sourcePath,
      fileType: ext.replace('.', ''),
      docIndex,
      page:
        doc.metadata?.loc?.pageNumber ||
        doc.metadata?.pageNumber ||
        null,
      uploadedAt: new Date().toISOString(),
    },
  }));

  // --- 第二步：切块 ---
  // chunkSize：每块最多包含多少字符
  // chunkOverlap：相邻两块之间重叠多少字符（避免语义在边界处被截断）
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 80,
    separators: ['\n\n', '\n', '。', '！', '？', '；', '，', ' ', ''],
  });
  const chunks = await splitter.splitDocuments(docsWithMetadata);
  const chunksWithMetadata = chunks.map((chunk, chunkId) => ({
    ...chunk,
    metadata: {
      ...chunk.metadata,
      chunkId,
      sourceId: `${storedFilename}#${chunkId}`,
    },
  }));
  console.log(`[RAG] 切块完成，共 ${chunks.length} 个块`);

  // --- 第三步：向量化 + 存入向量库 ---
  // 具体向量库细节由 vectorStore.js 处理。
  // 当前内部使用 pgvector，文档向量会持久化到 PostgreSQL。
  await addDocumentsToVectorStore(chunksWithMetadata);
  console.log(`[RAG] 向量化完成，已存入向量库`);

  return chunksWithMetadata.length;
}

/**
 * 删除某个上传文档对应的向量索引
 *
 * 文件删除和向量索引删除必须绑定：
 * 只删 uploads/ 文件会导致 RAG 仍然能从 pgvector 检索到旧内容。
 *
 * @param {string} storedFilename - 服务器保存的文件名
 * @returns {Promise<number>} 删除的 chunk 数量
 */
export async function deleteDocumentIndex(storedFilename) {
  return deleteDocumentsFromVectorStoreByStoredFilename(storedFilename);
}

/**
 * 清理 uploads/ 里已经不存在的文档对应的向量索引
 *
 * @param {string[]} validStoredFilenames - 当前仍存在的上传文档文件名
 * @returns {Promise<number>} 删除的孤儿 chunk 数量
 */
export async function cleanupMissingDocumentIndexes(validStoredFilenames) {
  return deleteDocumentIndexesNotInStoredFilenames(validStoredFilenames);
}

/**
 * 从向量库中检索与问题最相关的内容
 *
 * 这是 RAG 的在线阶段：
 *  1. 把用户问题向量化
 *  2. 在向量库中找距离最近的 k 个块（相似度最高）
 *  3. 返回这些块的文本内容，供 Agent 组装进 Prompt
 *
 * @param {string} query - 用户的问题
 * @param {number} k - 返回几个最相关的块，默认 3
 * @returns {{ context: string, sources: Array<object> }} 相关内容和来源引用
 */
export async function retrieveContext(query, k = 3) {
  if (!(await hasIndexedDocuments())) {
    return {
      context: '（当前没有上传任何文档，无法检索）',
      sources: [],
    };
  }

  // similaritySearch 返回最相似的 k 个 Document 对象
  const results = await searchSimilarDocuments(query, k);

  const sources = results.map((doc, index) => ({
    id: index + 1,
    filename: doc.metadata.filename || '未知文件',
    page: doc.metadata.page || null,
    chunkId: doc.metadata.chunkId,
    sourceId: doc.metadata.sourceId,
    sourcePath: doc.metadata.sourcePath,
    preview: doc.pageContent.slice(0, 120),
  }));

  // 把多个块的内容拼接成带来源编号的上下文，方便 Agent 在回答中引用。
  const context = results
    .map((doc, index) => {
      const source = sources[index];
      return [
        `[来源 ${source.id}]`,
        `文件：${source.filename}`,
        source.page ? `页码：${source.page}` : null,
        `chunkId：${source.chunkId}`,
        '',
        doc.pageContent,
      ]
        .filter(Boolean)
        .join('\n');
    })
    .join('\n\n---\n\n');

  return { context, sources };
}

/**
 * 判断当前是否已有文档加载进向量库
 *
 * Agent 可以用这个方法决定要不要调用 RAG 工具
 *
 * @returns {Promise<boolean>}
 */
export async function hasDocuments() {
  return hasIndexedDocuments();
}
