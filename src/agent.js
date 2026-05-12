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
// 每个会话用 thread_id 区分，相同 thread_id = 同一个对话。
//
// TODO：后续接入 LangSmith 可视化对话历史和调试轨迹
// -------------------------------------------------------
const checkpointer = SqliteSaver.fromConnString('./checkpoint.db');

// -------------------------------------------------------
// 创建 ReAct Agent
//
// createReactAgent 会自动构建一个循环图：
//   [用户输入] → LLM决策 → [调用工具?] → 工具执行 → LLM再决策 → [输出]
//
// stateModifier 用于在每次对话前注入系统提示词，
// 让 Agent 始终保持"私厨管家"的角色定位。
// -------------------------------------------------------
const agent = createAgent({
  model,                          // 对话模型
  tools: allTools,                // 工具集
  checkpointer,                   // 对话记忆（SQLite）
  systemPrompt: CHEF_SYSTEM_PROMPT, // 系统提示词（私厨角色定义）
});

/**
 * 调用 Agent 处理用户消息（一次性返回）
 *
 * @param {string} userMessage - 用户输入的文字
 * @param {string} threadId - 会话 ID，同一个 ID 会保留上下文记忆
 * @returns {string} Agent 的最终回答
 */
export async function invokeAgent(userMessage, threadId = 'default') {
  const config = {
    configurable: { thread_id: threadId },
  };

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
 * @returns {AsyncGenerator} 逐块产出 AI 回复文本
 */
export async function* streamAgent(userMessage, threadId = 'default') {
  const config = {
    configurable: { thread_id: threadId },
  };

  const stream = await agent.stream(
    {
      messages: [new HumanMessage(userMessage)],
    },
    {
      ...config,
      streamMode: 'messages',
    }
  );

  for await (const [message] of stream) {
    // 只取 AI 模型输出的文本块，跳过工具调用等非文本内容
    if (
      message._getType?.() === 'ai' &&
      message.content &&
      typeof message.content === 'string' &&
      !message.tool_calls?.length
    ) {
      yield message.content;
    }
  }
}
