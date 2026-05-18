// ============================================================
// vectorStore.js —— 向量存储适配层
//
// 这一层只暴露稳定能力：
//   addDocumentsToVectorStore()  存入文档向量
//   searchSimilarDocuments()     相似度检索
//   hasIndexedDocuments()        判断是否已有索引
//   deleteDocumentsFromVectorStoreByStoredFilename()
//                                删除某个上传文件对应的向量
//   deleteDocumentIndexesNotInStoredFilenames()
//                                清理 uploads/ 中已不存在的孤儿索引
//
// rag.js 不直接依赖具体向量数据库。
// 后续切换 Qdrant / Redis 时，优先改这个文件。
// ============================================================

import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { OpenAIEmbeddings } from '@langchain/openai';

// -------------------------------------------------------
// Embedding 模型初始化
//
// 文档入库时会调用 embedDocuments()
// 用户问题检索时会调用 embedQuery()
//
// 这里继续沿用你当前的 DashScope text-embedding-v3 配置。
// -------------------------------------------------------
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.EMBEDDING_API_KEY,
  model: process.env.EMBEDDING_MODEL || 'text-embedding-v3',
  configuration: {
    baseURL: process.env.EMBEDDING_BASE_URL,
  },
});

// pgvector 连接配置
//
// 这里的默认值和 docker-compose.yml 保持一致。
// 如果未来要连接云数据库，只需要改 .env，不需要改 RAG 流程代码。
const postgresConnectionOptions = {
  type: 'postgres',
  host: process.env.POSTGRES_HOST || '127.0.0.1',
  port: Number(process.env.POSTGRES_PORT || 5432),
  database: process.env.POSTGRES_DB || 'agent_demo',
  user: process.env.POSTGRES_USER || 'agent',
  password: process.env.POSTGRES_PASSWORD || 'agent_password',
};

const vectorStoreConfig = {
  postgresConnectionOptions,
  tableName: process.env.VECTOR_TABLE_NAME || 'rag_chunks',
  columns: {
    idColumnName: 'id',
    vectorColumnName: 'vector',
    contentColumnName: 'content',
    metadataColumnName: 'metadata',
  },
  // cosine 适合大多数文本 embedding 检索场景。
  distanceStrategy: 'cosine',
};

// PGVectorStore 初始化会连接数据库并确保表存在。
// 用 Promise 做单例，避免每次检索都重复建连接/建表。
let vectorStorePromise = null;

async function getVectorStore() {
  if (!vectorStorePromise) {
    vectorStorePromise = PGVectorStore.initialize(embeddings, vectorStoreConfig)
      .catch((error) => {
        // 初始化失败时清空单例，下一次请求可以重新尝试连接数据库。
        vectorStorePromise = null;
        throw error;
      });
  }

  return vectorStorePromise;
}

/**
 * 把切好的文档 chunk 写入向量库
 *
 * @param {Array} documents - LangChain Document 数组
 */
export async function addDocumentsToVectorStore(documents) {
  if (!documents || documents.length === 0) return;

  const vectorStore = await getVectorStore();
  // addDocuments 内部会自动调用 embeddings.embedDocuments()
  await vectorStore.addDocuments(documents);
}

/**
 * 根据用户问题做相似度检索
 *
 * @param {string} query - 用户问题
 * @param {number} k - 返回最相似的文档数量
 * @returns {Promise<Array>} 命中的 Document 数组
 */
export async function searchSimilarDocuments(query, k = 3) {
  const vectorStore = await getVectorStore();

  // similaritySearch 内部会自动调用 embeddings.embedQuery(query)
  return vectorStore.similaritySearch(query, k);
}

/**
 * 删除某个上传文件对应的所有 chunk 向量
 *
 * 删除上传文件时必须同步删向量索引。
 * 否则文件虽然从 uploads/ 消失，但 pgvector 里仍能检索到旧内容。
 *
 * @param {string} storedFilename - 服务器保存的文件名，例如 1779084707715.txt
 * @returns {Promise<number>} 删除的向量行数
 */
export async function deleteDocumentsFromVectorStoreByStoredFilename(storedFilename) {
  if (!storedFilename) return 0;

  const vectorStore = await getVectorStore();
  const metadataColumn = vectorStore.metadataColumnName;
  const result = await vectorStore.pool.query(
    `
      DELETE FROM ${vectorStore.computedTableName}
      WHERE "${metadataColumn}"->>'storedFilename' = $1
         OR "${metadataColumn}"->>'sourcePath' = $2
    `,
    [storedFilename, `uploads/${storedFilename}`]
  );

  return result.rowCount || 0;
}

/**
 * 删除已经没有对应上传文件的文档向量
 *
 * pgvector 是持久化存储，如果用户直接删了 uploads/ 文件，
 * 或者旧版本删除接口没有同步清理索引，就会留下“孤儿索引”。
 * 启动时用这个方法把向量表和 uploads/ 目录重新对齐。
 *
 * @param {string[]} validStoredFilenames - 当前 uploads/ 中真实存在的文档文件名
 * @returns {Promise<number>} 删除的孤儿向量行数
 */
export async function deleteDocumentIndexesNotInStoredFilenames(validStoredFilenames) {
  const vectorStore = await getVectorStore();
  const metadataColumn = vectorStore.metadataColumnName;

  if (!validStoredFilenames.length) {
    const result = await vectorStore.pool.query(
      `
        DELETE FROM ${vectorStore.computedTableName}
        WHERE "${metadataColumn}"->>'storedFilename' IS NOT NULL
      `
    );

    return result.rowCount || 0;
  }

  const result = await vectorStore.pool.query(
    `
      DELETE FROM ${vectorStore.computedTableName}
      WHERE "${metadataColumn}"->>'storedFilename' IS NOT NULL
        AND NOT (("${metadataColumn}"->>'storedFilename') = ANY($1::text[]))
    `,
    [validStoredFilenames]
  );

  return result.rowCount || 0;
}

/**
 * 判断 pgvector 表里是否已有文档向量
 *
 * 这里查数据库，而不是只看内存变量。
 * 原因：pgvector 是持久化存储，服务重启后表里仍然可能有历史文档。
 *
 * @returns {Promise<boolean>}
 */
export async function hasIndexedDocuments() {
  try {
    const vectorStore = await getVectorStore();
    const result = await vectorStore.pool.query(
      `SELECT 1 FROM ${vectorStore.computedTableName} LIMIT 1`
    );

    return result.rowCount > 0;
  } catch (error) {
    // 聊天功能不应该因为向量库暂时未启动而整体不可用。
    // 上传文档时仍会抛错，让用户知道文档没有成功入库。
    console.warn(`[VectorStore] 暂时无法检查文档索引: ${error.message}`);
    return false;
  }
}
