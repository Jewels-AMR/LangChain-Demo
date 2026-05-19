// ============================================================
// tools.js —— Agent 工具集
//
// 工具（Tool）是 Agent 能主动调用的能力模块。
// Agent 会根据用户问题自动判断要不要调用工具、调用哪个工具。
//
// 每个工具需要定义：
//  - name：工具名称（Agent 用这个名字来识别和调用）
//  - description：工具说明（Agent 读这段话来决定什么时候用它）
//  - schema：入参格式（用 Zod 定义，LangChain 会自动做校验）
//  - 函数体：实际执行逻辑
// ============================================================

import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';
import { tavily } from '@tavily/core';
import sharp from 'sharp';
import { retrieveContext, hasDocuments } from './rag.js';
import { findIndexedDocumentByIdentifier } from './documentStore.js';

function createDocumentSourcesArtifact({
  query,
  scopeType = 'all',
  documentIds = [],
  documentNames = [],
  sources = [],
  minSimilarityScore = null,
  candidateCount = 0,
  hitCount = sources.length,
  requestedK = 0,
  reason = null,
} = {}) {
  // artifact 是后端给前端看的结构化数据，不依赖模型自己总结。
  return {
    type: 'document_sources',
    query,
    scope: {
      type: scopeType,
      documentIds,
      names: documentNames,
    },
    sources,
    minSimilarityScore,
    candidateCount,
    hitCount,
    requestedK,
    reason,
  };
}

// -------------------------------------------------------
// 工具一：文档检索工具（RAG）
//
// 当用户问的问题可能涉及上传文件的内容时，
// Agent 会调用这个工具去向量库里找相关段落，
// 再把找到的内容结合进回答中。
// -------------------------------------------------------
export function createDocumentRetrievalTool(options = {}) {
  const forcedDocumentIds = Array.isArray(options.forcedDocumentIds)
    ? options.forcedDocumentIds.filter(Boolean)
    : [];
  const scopedDocuments = Array.isArray(options.scopedDocuments)
    ? options.scopedDocuments
    : [];

  return tool(
    async ({ query, documentId, filename }) => {
      if (!(await hasDocuments())) {
        return [
          '用户还没有上传任何文档，请提示用户先上传文件。',
          createDocumentSourcesArtifact({
            query,
            scopeType: 'none',
            sources: [],
            reason: 'no_documents',
          }),
        ];
      }

      const targetDocument = documentId
        ? await findIndexedDocumentByIdentifier(documentId)
        : filename
          ? await findIndexedDocumentByIdentifier(filename)
          : null;
      const forcedIdSet = new Set(forcedDocumentIds);

      if ((documentId || filename) && !targetDocument) {
        return [
          `没有找到可检索的指定文档：${documentId || filename}。请改为检索全部已上传文档，或提示用户确认文件名。`,
          createDocumentSourcesArtifact({
            query,
            scopeType: forcedDocumentIds.length > 0 ? 'selected' : 'all',
            documentIds: forcedDocumentIds,
            documentNames: scopedDocuments.map((doc) => doc.originalName),
            sources: [],
            reason: 'target_not_found',
          }),
        ];
      }

      if (
        forcedDocumentIds.length > 0 &&
        targetDocument &&
        !forcedIdSet.has(targetDocument.id)
      ) {
        const scopeNames = scopedDocuments.map((doc) => doc.originalName).join('、');
        return [
          `当前会话已限定检索范围：${scopeNames || '选中文档'}。指定文档不在当前范围内，不能检索。`,
          createDocumentSourcesArtifact({
            query,
            scopeType: 'selected',
            documentIds: forcedDocumentIds,
            documentNames: scopedDocuments.map((doc) => doc.originalName),
            sources: [],
            reason: 'outside_scope',
          }),
        ];
      }

      const documentIds = forcedDocumentIds.length > 0
        ? (targetDocument ? [targetDocument.id] : forcedDocumentIds)
        : (targetDocument ? [targetDocument.id] : []);
      const documentNames = targetDocument
        ? [targetDocument.originalName]
        : scopedDocuments.map((doc) => doc.originalName);

      const {
        context,
        sources,
        minSimilarityScore,
        candidateCount,
        hitCount,
        requestedK,
      } = await retrieveContext(query, 3, {
        documentIds,
      });
      const scopeText = documentIds.length > 0
        ? `检索范围：${
            targetDocument
              ? targetDocument.originalName
              : scopedDocuments.map((doc) => doc.originalName).join('、')
          }\n\n`
        : '';
      const reason = sources.length > 0 ? null : 'no_relevant_chunks';

      // content 给模型阅读，用来生成回答；artifact 给后端/前端使用，不交给模型自由改写。
      return [
        `${scopeText}检索候选数：${candidateCount}，命中数：${hitCount}，最低相关度阈值：${minSimilarityScore.toFixed(2)}。\n\n以下是从文档中检索到的相关内容：\n\n${context}\n\n请基于上述文档内容回答。引用来源由系统单独展示，你不要在回答末尾重复手写来源列表。如果没有检索到高相关片段，请直接说明文档中没有找到足够相关的信息。`,
        createDocumentSourcesArtifact({
          query,
          scopeType: documentIds.length > 0 ? 'selected' : 'all',
          documentIds,
          documentNames,
          sources,
          minSimilarityScore,
          candidateCount,
          hitCount,
          requestedK,
          reason,
        }),
      ];
    },
    {
      name: 'document_retrieval',
      description:
        '当用户的问题涉及他上传的文件内容时使用此工具。' +
        '输入用户的问题，工具会从上传的文档中检索最相关的段落并返回。' +
        '如果用户明确指定某个文件，请传入 documentId；如果只有文件名，请传入 filename。',
      responseFormat: 'content_and_artifact',
      schema: z.object({
        query: z.string().describe('用户的问题或搜索关键词'),
        documentId: z.string().optional().describe('可选，指定只检索某个 documents.id'),
        filename: z.string().optional().describe('可选，指定只检索某个上传文件名或原始文件名'),
      }),
    }
  );
}

export const documentRetrievalTool = createDocumentRetrievalTool();

// -------------------------------------------------------
// 工具二：图片分析工具（多模态）
//
// 工作原理：
//  1. 读取图片文件，转成 Base64 字符串
//  2. 构造包含图片的多模态消息（文字 + 图片）
//  3. 发给视觉模型（Vision LLM），得到识别结果
//
// Base64：把二进制图片数据编码成纯文字字符串的格式，
// 方便通过 JSON/HTTP 传输给 API。
//
// 多模态消息格式（LangChain 标准）：
// [{ type: 'text', text: '...' }, { type: 'image_url', image_url: { url: 'data:image/...' } }]
// -------------------------------------------------------

// 视觉模型单独初始化，和对话模型分开
// 因为不是所有模型都支持视觉，需要单独指定支持图片的模型
const visionModel = new ChatOpenAI({
  // 从环境变量读取视觉模型名称，默认使用 gpt-4o
  // TODO：如果使用 DeepSeek 视觉模型，在 .env 中设置 VISION_MODEL=deepseek-vl2
  model: process.env.VISION_MODEL || 'gpt-4o',
  apiKey: process.env.VISION_API_KEY || process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.VISION_BASE_URL || 'https://api.openai.com/v1',
  },
});

export const imageAnalysisTool = tool(
  async ({ imagePath, question }) => {
    console.log(`[Tool] 图片分析工具被调用，图片路径: ${imagePath}`);

    // 第一步：压缩图片（视觉模型对文件大小有限制，通常 5MB 以内）
    // sharp 会把图片缩放到最大 1024px，质量 80%，统一输出 jpeg
    const compressedBuffer = await sharp(imagePath)
      .resize(1024, 1024, { fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 80 })
      .toBuffer();

    const base64Image = compressedBuffer.toString('base64');
    console.log(`[Tool] 图片压缩完成，大小: ${(compressedBuffer.length / 1024).toFixed(0)}KB`);

    // 第二步：构造多模态消息（文字 + 图片）
    const message = new HumanMessage({
      content: [
        {
          type: 'text',
          text: question || '请识别图片中所有可见的食材，评估新鲜度，整理成"可用食材清单"。',
        },
        {
          type: 'image_url',
          image_url: {
            url: `data:image/jpeg;base64,${base64Image}`,
          },
        },
      ],
    });

    // 第四步：调用视觉模型，获取分析结果
    const response = await visionModel.invoke([message]);
    return response.content;
  },
  {
    name: 'image_analysis',
    description:
      '当用户上传了图片并希望分析图片内容时使用此工具。' +
      '可以识别图片中的食材、菜品，并根据识别结果推荐菜谱或烹饪建议。',
    schema: z.object({
      imagePath: z.string().describe('图片文件在服务器上的路径'),
      question: z.string().describe('用户关于图片的问题，例如：这些食材能做什么菜？'),
    }),
  }
);

// -------------------------------------------------------
// 工具三：联网搜索工具（Tavily）
//
// Tavily 是专门为 AI Agent 设计的搜索引擎：
//  - 返回结构化的搜索结果（而不是原始 HTML）
//  - 支持过滤新闻、学术等来源
//  - 比 Google/Bing 更适合 LLM 直接消费
//
// 使用场景：
//  - 搜索某种食材的详细菜谱
//  - 查询某道菜的做法、营养价值
//  - 根据识别出的食材在网上找灵感
//
// 需要在 .env 中设置：TAVILY_API_KEY=tvly-xxxxxx
// 去 https://tavily.com 免费注册获取
// -------------------------------------------------------
// tavily() 是工厂函数，传入 apiKey 返回客户端实例
const tavilyClient = tavily({ apiKey: process.env.TAVILY_API_KEY });

function isHttpUrl(value) {
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol);
  } catch {
    return false;
  }
}

function formatSearchImages(images = []) {
  const validImages = images
    .map((image) => ({
      url: typeof image === 'string' ? image : image?.url,
      description: typeof image === 'string' ? '' : image?.description,
    }))
    .filter((image) => image.url && isHttpUrl(image.url))
    .slice(0, 5);

  if (validImages.length === 0) {
    return '可用参考图片：无。请不要输出图片占位符、base64、二进制内容或无法访问的图片链接。';
  }

  // 只把干净的 http(s) URL 交给模型，避免模型把图片二进制/base64 当成正文输出。
  return [
    '可用参考图片（只能从下面的 URL 中选择，禁止输出 base64、二进制内容或乱码）：',
    ...validImages.map((image, index) =>
      `[图片 ${index + 1}] ${image.description || '参考图片'}\n图片URL：${image.url}\nMarkdown：![参考图片](${image.url})`
    ),
  ].join('\n');
}

export const webSearchTool = tool(
  async ({ query }) => {
    // 天气类查询自动补充今天日期，避免 Tavily 返回缓存的过期数据
    const weatherKeywords = ['天气', '气温', '温度', '下雨', '晴', '阴', '风', 'weather', 'forecast'];
    const isWeatherQuery = weatherKeywords.some(k => query.includes(k));
    const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
    const finalQuery = isWeatherQuery ? `${query} ${today}` : query;

    console.log(`[Tool] 联网搜索: ${finalQuery}`);

    // search() 发起搜索，返回若干条结果。
    // includeImages 会额外返回图片 URL，供 Agent 在菜谱报告里插入 Markdown 图片。
    const results = await tavilyClient.search(finalQuery, {
      maxResults: 5,
      includeImages: true,
      includeImageDescriptions: true,
    });

    // 把搜索结果格式化成易于 LLM 阅读的文本
    // 每条结果包含：标题、URL、摘要
    const formatted = results.results
      .map((r, i) => `[${i + 1}] ${r.title}\n来源：${r.url}\n摘要：${r.content}`)
      .join('\n\n');

    const imageReferences = formatSearchImages(results.images);

    return `以下是联网搜索"${query}"的结果：\n\n${formatted}\n\n${imageReferences}`;
  },
  {
    name: 'web_search',
    description:
      '当需要搜索菜谱、食材信息、烹饪技巧或其他需要最新网络信息时使用此工具。' +
      '例如：搜索某道菜的做法、查询食材的营养价值、找创意菜谱灵感。',
    schema: z.object({
      query: z.string().describe('搜索关键词，例如：番茄炒蛋家常做法、三文鱼菜谱推荐'),
    }),
  }
);

// 导出所有工具，供 Agent 注册使用
export const allTools = [documentRetrievalTool, imageAnalysisTool, webSearchTool];
