// ============================================================
// agent.js —— LangGraph Agent 核心
//
// Agent 是整个系统的大脑：
//  - 接收用户消息
//  - 决定是否需要调用工具（RAG检索、图片分析等）
//  - 调用工具后观察结果，再决定下一步
//  - 最终生成回答
//
// 这里使用 ReAct 模式（Reasoning + Acting）：
//  思考 → 行动（调用工具）→ 观察结果 → 再思考 → 最终回答
// ============================================================

import { ChatOpenAI } from '@langchain/openai';
import { createAgent } from 'langchain';
import { SqliteSaver } from '@langchain/langgraph-checkpoint-sqlite';
import { HumanMessage } from '@langchain/core/messages';
import { CHEF_SYSTEM_PROMPT } from './prompts.js';
import { allTools } from './tools.js';
import { hasDocuments } from './rag.js';

// -------------------------------------------------------
// LLM 模型初始化
//
// 这里使用 DeepSeek，通过兼容 OpenAI 接口的方式接入。
// 后续切换为多模态模型时，只需修改 model 名称和 baseURL。
//
// TODO：多模态图片分析阶段，替换为支持视觉的模型
//（如 deepseek-vl2 或接入 OpenAI gpt-4o）
// -------------------------------------------------------
const model = new ChatOpenAI({
  model: process.env.LLM_MODEL || 'deepseek-chat',
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL || 'https://api.deepseek.com/v1',
  },
  temperature: 0,
  // DeepSeek 思考模式（Thinking Mode）在工具调用场景下需要特殊处理：
  // 模型输出的 reasoning_content 必须回传，但 LangChain 暂不支持。
  // 通过 modelKwargs 传入 extra_body，关闭思考模式，确保工具调用正常运行。
  modelKwargs: {
    thinking: { type: 'disabled' },
  },
});

// -------------------------------------------------------
// Checkpoint（对话记忆）初始化
//
// Checkpoint 让 Agent 能"记住"之前的对话。
// 使用 SQLite 把对话历史持久化到本地文件。
//
// TODO：后续接入 LangSmith 可视化对话历史和调试轨迹
// -------------------------------------------------------
const checkpointer = SqliteSaver.fromConnString('./checkpoint.db');

const toolLabels = {
  document_retrieval: '检索上传文档',
  image_analysis: '分析图片内容',
  web_search: '联网搜索信息',
};

// 把工具英文名转成前端可读的中文步骤文案。
function getToolLabel(toolName) {
  return toolLabels[toolName] || toolName || '工具调用';
}

// 同一次 Agent loop 中，模型可能分多次流式吐出同一个 tool call。
// 这里生成稳定 key，用来避免前端重复显示同一个工具步骤。
function getToolCallKey(toolCall) {
  return toolCall.id || `${toolCall.name}:${toolCall.index ?? ''}`;
}

// 工具结果可能很长，只截取一小段作为调试预览，避免前端步骤面板被撑爆。
function summarizeToolContent(content) {
  const text = typeof content === 'string'
    ? content
    : JSON.stringify(content);

  return text.replace(/\s+/g, ' ').slice(0, 120);
}

function selectTools({ hasImage, hasDocs }) {
  return allTools.filter((tool) => {
    // 没有图片时不把 image_analysis 暴露给模型，避免模型误判后进入图片分析流程。
    if (tool.name === 'image_analysis') return hasImage;

    // 没有文档索引时不暴露 document_retrieval，普通菜谱问题直接走 web_search。
    if (tool.name === 'document_retrieval') return hasDocs;

    return true;
  });
}

function buildSystemPrompt({ hasImage, hasDocs }) {
  const requestState = [
    '【当前请求状态】',
    hasImage
      ? '- 本轮用户已上传图片，可以在需要时调用 image_analysis。'
      : '- 本轮用户没有上传图片，禁止调用或提及图片分析，除非用户明确询问图片状态。',
    hasDocs
      ? '- 当前已有上传文档索引，可以在文档相关问题中调用 document_retrieval。'
      : '- 当前没有可检索的上传文档。普通菜谱、天气、搭配建议等问题不要说明"没有上传文档"，直接继续处理；只有用户明确询问文档内容时才说明目前没有文档。',
    '- 不要在回答开头描述"我先检查是否有图片/文档"这类内部流程，直接给用户结果。',
  ].join('\n');

  return `${CHEF_SYSTEM_PROMPT}\n\n${requestState}`;
}

async function createChefAgent(options = {}) {
  const hasImage = Boolean(options.hasImage);
  const hasDocs = await hasDocuments();
  const tools = selectTools({ hasImage, hasDocs });

  // 每次请求按上下文创建 Agent：
  // 工具集会随“是否有图片/是否有文档”变化，Agent loop 就不会走不存在的分支。
  return {
    agent: createAgent({
      model,
      tools,
      checkpointer,
      systemPrompt: buildSystemPrompt({ hasImage, hasDocs }),
    }),
    toolNames: new Set(tools.map((tool) => tool.name)),
  };
}

/**
 * 调用 Agent 处理用户消息（一次性返回）
 *
 * @param {string} userMessage - 用户输入的文字
 * @param {string} threadId - 会话 ID，同一个 ID 会保留上下文记忆
 * @returns {string} Agent 的最终回答
 */
export async function invokeAgent(userMessage, threadId = 'default', options = {}) {
  const config = {
    configurable: { thread_id: threadId },
  };
  const { agent } = await createChefAgent(options);

  const response = await agent.invoke(
    {
      messages: [new HumanMessage(userMessage)],
    },
    config
  );

  const lastMessage = response.messages.at(-1);
  return lastMessage.content;
}

/**
 * 流式调用 Agent 处理用户消息
 *
 * @param {string} userMessage - 用户输入的文字
 * @param {string} threadId - 会话 ID
 * @returns {AsyncGenerator} 逐块产出 Agent 事件
 */
export async function* streamAgent(userMessage, threadId = 'default', options = {}) {
  const config = {
    configurable: { thread_id: threadId },
  };
  const { agent, toolNames: activeToolNames } = await createChefAgent(options);

  const stream = await agent.stream(
    {
      messages: [new HumanMessage(userMessage)],
    },
    {
      ...config,
      streamMode: 'messages',
    }
  );

  const startedTools = new Set();
  const finishedTools = new Set();
  const toolCallNames = new Map();

  for await (const [message] of stream) {
    const messageType = message._getType?.();

    // AI 消息分两类：
    // 1. tool_calls / tool_call_chunks：表示模型决定调用某个工具
    // 2. content：表示模型正在输出最终回答文本
    if (messageType === 'ai') {
      const toolCalls = message.tool_calls || [];
      const toolCallChunks = (message.tool_call_chunks || []).filter((chunk) =>
        chunk.name && activeToolNames.has(chunk.name)
      );
      const toolCandidates = [...toolCalls, ...toolCallChunks].filter((toolCall) =>
        toolCall.name && activeToolNames.has(toolCall.name)
      );

      for (const toolCall of toolCandidates) {
        const key = getToolCallKey(toolCall);
        toolCallNames.set(key, toolCall.name);

        if (!startedTools.has(key)) {
          startedTools.add(key);
          const label = getToolLabel(toolCall.name);
          yield {
            type: 'tool_start',
            id: key,
            tool: toolCall.name,
            label,
            message: `正在${label}...`,
          };
        }
      }

      // 只把最终回答的文本块发给前端；工具调用消息由上面的事件单独表示。
      if (
        message.content &&
        typeof message.content === 'string' &&
        !toolCandidates.length
      ) {
        yield {
          type: 'text',
          content: message.content,
        };
      }
    }

    // ToolMessage 表示工具已经执行完成。
    // 这里把工具完成事件也发给前端，前端就能把“正在...”更新成“完成/失败”。
    if (messageType === 'tool') {
      const key = message.tool_call_id || message.id || message.name || 'unknown_tool';
      if (finishedTools.has(key)) continue;

      finishedTools.add(key);
      const toolName = toolCallNames.get(key) || message.name || 'unknown_tool';
      if (!activeToolNames.has(toolName)) continue;
      const label = getToolLabel(toolName);
      const isError = message.status === 'error';

      yield {
        type: 'tool_result',
        id: key,
        tool: toolName,
        label,
        status: isError ? 'error' : 'success',
        message: isError ? `${label}失败` : `${label}完成`,
        preview: summarizeToolContent(message.content || ''),
      };

      // document_retrieval 会把命中的文档来源放在 ToolMessage.artifact。
      // 这部分是检索系统的事实数据，单独发给前端渲染，避免让模型手写引用时出错。
      if (
        toolName === 'document_retrieval' &&
        message.artifact?.type === 'document_sources' &&
        Array.isArray(message.artifact.sources) &&
        message.artifact.sources.length > 0
      ) {
        yield {
          type: 'sources',
          sources: message.artifact.sources,
        };
      }
    }
  }
}
