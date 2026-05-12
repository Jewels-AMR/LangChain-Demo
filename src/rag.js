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
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory';
import { OpenAIEmbeddings } from '@langchain/openai';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { TextLoader } from '@langchain/classic/document_loaders/fs/text';
import path from 'path';

// -------------------------------------------------------
// Embedding 模型初始化
//
// Embedding 模型负责把文字转成向量（一串数字）。
// DeepSeek 没有 Embedding 模型，这里使用阿里云 Dashscope 的
// text-embedding-v3，通过兼容 OpenAI 的接口调用。
//
// 向量维度：text-embedding-v3 输出 1024 维向量
// 计费：按 token 计费，通常很便宜
// -------------------------------------------------------
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.EMBEDDING_API_KEY,
  model: process.env.EMBEDDING_MODEL || 'text-embedding-v3',
  configuration: {
    baseURL: process.env.EMBEDDING_BASE_URL,
  },
});

// -------------------------------------------------------
// 向量库（内存版）
//
// MemoryVectorStore：把向量存在内存里，重启后消失。
// 优点：零配置，适合开发阶段。
// 缺点：不持久化。
//
// TODO：后续替换为 Chroma 等持久化向量库
// -------------------------------------------------------
let vectorStore = null;

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
 * @returns {number} 最终存入向量库的块数量
 */
export async function processDocument(filePath) {
  // --- 第一步：加载文件 ---
  const loader = getLoader(filePath);
  const docs = await loader.load();
  console.log(`[RAG] 文件加载完成，共 ${docs.length} 个文档段`);

  // --- 第二步：切块 ---
  // chunkSize：每块最多包含多少字符
  // chunkOverlap：相邻两块之间重叠多少字符（避免语义在边界处被截断）
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const chunks = await splitter.splitDocuments(docs);
  console.log(`[RAG] 切块完成，共 ${chunks.length} 个块`);

  // --- 第三步：向量化 + 存入向量库 ---
  // fromDocuments 内部会自动调用 embeddings.embedDocuments()
  // 把每个 chunk 的文本发给 Embedding 模型，拿回向量，存入内存
  if (!vectorStore) {
    vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);
  } else {
    await vectorStore.addDocuments(chunks);
  }
  console.log(`[RAG] 向量化完成，已存入向量库`);

  return chunks.length;
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
 * @returns {string} 拼接好的相关内容文本
 */
export async function retrieveContext(query, k = 3) {
  if (!vectorStore) {
    return '（当前没有上传任何文档，无法检索）';
  }

  // similaritySearch 返回最相似的 k 个 Document 对象
  const results = await vectorStore.similaritySearch(query, k);

  // 把多个块的内容拼接成一段文字，方便塞进 Prompt
  return results.map((doc) => doc.pageContent).join('\n\n---\n\n');
}

/**
 * 判断当前是否已有文档加载进向量库
 *
 * Agent 可以用这个方法决定要不要调用 RAG 工具
 *
 * @returns {boolean}
 */
export function hasDocuments() {
  return vectorStore !== null;
}
