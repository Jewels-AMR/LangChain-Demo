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

// -------------------------------------------------------
// 工具一：文档检索工具（RAG）
//
// 当用户问的问题可能涉及上传文件的内容时，
// Agent 会调用这个工具去向量库里找相关段落，
// 再把找到的内容结合进回答中。
// -------------------------------------------------------
export const documentRetrievalTool = tool(
  async ({ query }) => {
    if (!hasDocuments()) {
      return '用户还没有上传任何文档，请提示用户先上传文件。';
    }
    const context = await retrieveContext(query);
    return `以下是从文档中检索到的相关内容：\n\n${context}`;
  },
  {
    name: 'document_retrieval',
    description:
      '当用户的问题涉及他上传的文件内容时使用此工具。' +
      '输入用户的问题，工具会从上传的文档中检索最相关的段落并返回。',
    schema: z.object({
      query: z.string().describe('用户的问题或搜索关键词'),
    }),
  }
);

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

export const webSearchTool = tool(
  async ({ query }) => {
    // 天气类查询自动补充今天日期，避免 Tavily 返回缓存的过期数据
    const weatherKeywords = ['天气', '气温', '温度', '下雨', '晴', '阴', '风', 'weather', 'forecast'];
    const isWeatherQuery = weatherKeywords.some(k => query.includes(k));
    const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
    const finalQuery = isWeatherQuery ? `${query} ${today}` : query;

    console.log(`[Tool] 联网搜索: ${finalQuery}`);

    // search() 发起搜索，返回若干条结果
    // maxResults：最多返回几条，太多会占用过多 token
    const results = await tavilyClient.search(finalQuery, { maxResults: 5 });

    // 把搜索结果格式化成易于 LLM 阅读的文本
    // 每条结果包含：标题、URL、摘要
    const formatted = results.results
      .map((r, i) => `[${i + 1}] ${r.title}\n来源：${r.url}\n摘要：${r.content}`)
      .join('\n\n');

    return `以下是联网搜索"${query}"的结果：\n\n${formatted}`;
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
