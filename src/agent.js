// ============================================================
// agent.js —— LangGraph 多 Agent 核心
//
// 本文件是整个系统的大脑，集成了多种 LangGraph 核心模式：
//
// 【多 Agent（Supervisor + Sub-graph）】
//  - supervisor 总调度决定分派给哪个专家
//  - chef_agent 烹饪专家子图（菜谱/购物/营养/天气工具）
//  - research_agent 检索专家子图（文档检索/联网搜索/图片工具）
//  - 专家完成后回到 supervisor，可继续分派（循环协作）
//
// 【Send 并行扇出】
//  - meal_planner 识别多道菜后，Send × N 并行研究
//  - dishReports 通过 concat reducer 自动合并结果
//
// 【Annotation 自定义状态】
//  - ChefState 扩展 MessagesAnnotation，增加 intent / nextAgent / dishReports 等字段
//
// 【图的完整运行路径】
//
//  START → intent_classifier
//    ├─ off_topic     → gentle_refuse → END
//    ├─ general_chat  → supervisor ↔ 专家子图（循环）→ END
//    └─ meal_plan     → meal_planner → [Send × N] → meal_aggregator → END
//
// 本文件负责的完整链路：
//
//  1. server.js 调用 streamAgent(message, threadId, options)
//  2. createChefAgent() 根据本轮上下文动态选择工具
//     - 有文档：暴露 document_retrieval
//     - 有图片：暴露 image_analysis
//     - 普通任务：暴露菜谱、购物清单、营养、天气、搜索工具
//  3. buildSystemPrompt() 生成本轮系统提示词
//  4. buildChefGraph() 构建多 Agent LangGraph
//     - 工具按专家分组，构建两个专家子图
//     - 子图嵌入父图，supervisor 协调调度
//  5. graph.stream() 运行，streamAgent() 把输出转成前端 SSE 事件：
//     - text：最终回答文本（来自专家子图 / meal_aggregator / gentle_refuse）
//     - tool_start / tool_result：工具调用事件（来自子图内部 ToolNode）
//     - rag_debug / sources：RAG 检索调试信息
// ============================================================

// ChatOpenAI 是 LangChain 对 OpenAI-compatible chat model 的封装。
// 这里虽然类名叫 ChatOpenAI，但 baseURL 指向 DeepSeek 时也可以用。
import { ChatOpenAI } from '@langchain/openai';

// LangGraph 基础构件：
// - Annotation：定义自定义状态字段和 reducer（合并策略）
// - StateGraph：声明”节点 + 边 + 状态”的图
// - MessagesAnnotation：预置 messages 状态，并自动合并新消息
// - Send：动态并行扇出（fan-out），把状态发送到多个并行节点
// - START / END：图的开始和结束保留节点
import {
  Annotation,
  END,
  MessagesAnnotation,
  Send,
  START,
  StateGraph,
} from '@langchain/langgraph';

// ToolNode 是 LangGraph 预置节点：
// 它会读取上一条 AIMessage 里的 tool_calls，并执行对应工具。
// toolsCondition 是预置条件函数：
// 它判断上一条 AIMessage 是否包含 tool_calls，决定去 tools 还是 END。
import { ToolNode, toolsCondition } from '@langchain/langgraph/prebuilt';
import { SqliteSaver } from '@langchain/langgraph-checkpoint-sqlite';

// HumanMessage / SystemMessage 是 LangChain 标准消息类型。
// HumanMessage：用户输入
// SystemMessage：系统提示词，控制 Agent 行为边界和工具选择规则
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import {
  CHEF_SYSTEM_PROMPT,
  SUPERVISOR_PROMPT,
  INTENT_CLASSIFIER_PROMPT,
  MEAL_PLANNER_PROMPT,
  createDishResearcherPrompt,
  MEAL_AGGREGATOR_PROMPT,
  GENTLE_REFUSE_PROMPT,
} from './prompts.js';

// tools.js 中的每个 tool 都是 LangChain StructuredTool。
// 模型看到的是 tool name / description / schema，
// 真正执行工具是在 LangGraph 的 ToolNode 中完成。
import {
  allTools,
  createDocumentRetrievalTool,
  imageAnalysisTool,
  nutritionEstimatorTool,
  recipePlannerTool,
  shoppingListTool,
  weatherLookupTool,
  webSearchTool,
} from './tools.js';
import { hasDocuments } from './rag.js';
import { listIndexedDocuments } from './documentStore.js';

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

// -------------------------------------------------------
// 自定义状态注解（Annotation）
//
// 标准 MessagesAnnotation 只有一个 messages 字段。
// 增强版图需要在节点之间传递更多信息：
//
// 【意图分类相关】
// - intent：意图分类结果，决定路由走哪条分支
//
// 【Supervisor 多 Agent 相关】
// - nextAgent：supervisor 的分派决定，指向下一个专家子图
// - delegationCount：supervisor 已分派次数，用于防止无限循环
//
// 【Send 并行相关】
// - plannedDishes：菜品规划器输出的菜名列表
// - currentDish：Send 并行时，每个分支处理的当前菜名
// - dishReports：并行研究的结果集合
//
// 每个字段需要指定 reducer（合并策略）和 default（初始值）。
//
// reducer 决定"当多个节点同时更新同一个字段时怎么合并"：
// - (_, val) => val：后写入的覆盖前面的（last-write-wins）
// - (prev, next) => prev.concat(next)：追加合并（用于 Send 并行结果）
//
// dishReports 的 concat reducer 是 Send 并行模式的关键：
// 每个 dish_researcher 分支返回 [{ dish, report }]，
// LangGraph 自动用 reducer 把多个分支的结果合并成一个完整数组。
// -------------------------------------------------------
const ChefState = Annotation.Root({
  // 继承 MessagesAnnotation 的 messages 字段及其 reducer。
  // 这样 agent / supervisor 子图等节点返回 { messages } 时，
  // 消息会自动合并进对话历史。
  ...MessagesAnnotation.spec,

  // 意图分类结果。由 intent_classifier 节点写入。
  // 后续条件边 routeByIntent 读取此字段决定路由。
  intent: Annotation({
    reducer: (_, val) => val,
    default: () => 'general_chat',
  }),

  // Supervisor 的分派决定。
  // 每次 supervisor 运行后写入 'chef_agent' / 'research_agent' / 'done'。
  // routeFromSupervisor 读取此字段决定把任务交给哪个专家子图。
  nextAgent: Annotation({
    reducer: (_, val) => val,
    default: () => 'done',
  }),

  // Supervisor 已分派次数。
  // 防止 supervisor ↔ 专家 之间形成无限循环。
  // 每次 supervisor 运行时 +1，超过上限（3 次）强制结束。
  //
  // 为什么需要这个字段：
  // supervisor 循环（supervisor → 专家 → supervisor → ...）是设计中的正常行为，
  // 但如果模型持续不返回 'done'，循环就不会终止。
  // delegationCount 是兜底保护。
  delegationCount: Annotation({
    reducer: (_, val) => val,
    default: () => 0,
  }),

  // 菜品规划器识别的候选菜品列表。
  // 由 meal_planner 节点写入，routeFromPlanner 读取后用 Send 扇出。
  plannedDishes: Annotation({
    reducer: (_, val) => val,
    default: () => [],
  }),

  // Send 并行时每个分支处理的单道菜名。
  // 由 Send 构造时传入，dish_researcher 节点读取。
  // 每个并行分支拿到的 currentDish 值不同。
  currentDish: Annotation({
    reducer: (_, val) => val,
    default: () => '',
  }),

  // 并行研究的结果数组。
  // 每个 dish_researcher 返回 [{ dish, report }]，
  // concat reducer 把所有并行分支的结果自动拼接成完整列表。
  // meal_aggregator 读取此字段生成最终推荐方案。
  dishReports: Annotation({
    reducer: (prev, next) => prev.concat(next),
    default: () => [],
  }),
});

// 工具名 -> 前端显示文案。
//
// 模型使用的是英文工具名，比如 recipe_planner；
// 前端用户更适合看到中文步骤，比如“规划菜谱方案”。
// streamAgent() 里会用这个映射生成 tool_start/tool_result 事件。
const toolLabels = {
  document_retrieval: '检索上传文档',
  image_analysis: '分析图片内容',
  recipe_planner: '规划菜谱方案',
  shopping_list: '整理购物清单',
  nutrition_estimator: '估算营养热量',
  weather_lookup: '查询天气信息',
  web_search: '联网搜索信息',
};

/**
 * 把工具英文名转成前端可读的中文步骤文案。
 *
 * @param {string} toolName - LangChain tool.name
 * @returns {string} 前端步骤面板里展示的中文名
 */
function getToolLabel(toolName) {
  return toolLabels[toolName] || toolName || '工具调用';
}

/**
 * 为 tool call 生成稳定 key。
 *
 * 为什么需要它：
 * - LangGraph streamMode=messages 时，模型可能用多个 chunk 流式吐出同一个 tool call。
 * - 如果每个 chunk 都渲染一次，前端会重复出现“正在调用工具”。
 * - tool_call.id 是最稳定的；如果没有 id，就用 name + index 兜底。
 *
 * @param {object} toolCall - AIMessage.tool_calls 或 tool_call_chunks 中的一项
 * @returns {string} 前端步骤去重用的 key
 */
function getToolCallKey(toolCall) {
  return toolCall.id || `${toolCall.name}:${toolCall.index ?? ''}`;
}

/**
 * 从 tool call 中读取工具入参。
 *
 * 注意：
 * - 流式 tool_call_chunk 初期可能只有工具名，还没有完整 args。
 * - 所以这里必须判断 args 是否是非空对象。
 * - 后续 tool_result 事件会尽量把 args 补给前端，方便学习 Agent 调了什么参数。
 *
 * @param {object} toolCall - 模型输出的工具调用描述
 * @returns {object | undefined} 工具入参
 */
function getToolCallArgs(toolCall) {
  return toolCall.args &&
    typeof toolCall.args === 'object' &&
    Object.keys(toolCall.args).length > 0
    ? toolCall.args
    : undefined;
}

/**
 * 压缩工具输出，给前端步骤 tooltip 做预览。
 *
 * 工具结果可能很长，例如 web_search 会返回多个网页摘要，
 * document_retrieval 会返回多个 chunk 内容。
 * 前端步骤面板只需要一个短预览，完整内容仍交给模型继续推理。
 *
 * @param {string | object} content - ToolMessage.content
 * @returns {string} 120 字以内的单行预览
 */
function summarizeToolContent(content) {
  const text = typeof content === 'string'
    ? content
    : JSON.stringify(content);

  return text.replace(/\s+/g, ' ').slice(0, 120);
}

/**
 * 把 RAG 工具 artifact 转成前端 rag_debug 事件。
 *
 * document_retrieval 工具返回两类信息：
 * - content：给模型看的文本，上下文会进入下一轮 agent 推理
 * - artifact：给系统/前端看的结构化数据，不让模型改写
 *
 * 这里读取 artifact，生成 rag_debug：
 * - query：真正用于检索的 query
 * - scope：检索范围，是全部文档还是选中文档
 * - candidateCount：向量库返回的候选 chunk 数
 * - hitCount：通过相似度阈值的 chunk 数
 * - sources：结构化引用来源
 *
 * @param {string} toolCallId - 当前 document_retrieval 的 tool_call_id
 * @param {object} artifact - document_retrieval 的结构化结果
 * @returns {object} SSE rag_debug 事件
 */
function createRagDebugEvent(toolCallId, artifact) {
  const sources = Array.isArray(artifact.sources) ? artifact.sources : [];
  const scope = artifact.scope && typeof artifact.scope === 'object'
    ? artifact.scope
    : { type: 'all', documentIds: [], names: [] };

  // 这条事件给前端展示“检索发生了什么”，不是给模型继续推理。
  return {
    // 前端根据 type === 'rag_debug' 渲染“检索过程”折叠面板。
    type: 'rag_debug',
    // id 用来和 tool_start/tool_result 对齐，属于同一次工具调用。
    id: toolCallId,
    // query 是工具实际检索的文本，可能和用户原话不完全一样。
    query: artifact.query || '',
    scope: {
      type: scope.type || 'all',
      documentIds: Array.isArray(scope.documentIds) ? scope.documentIds : [],
      names: Array.isArray(scope.names) ? scope.names : [],
    },
    candidateCount: Number(artifact.candidateCount || 0),
    hitCount: Number(artifact.hitCount ?? sources.length),
    requestedK: Number(artifact.requestedK || 0),
    minSimilarityScore:
      typeof artifact.minSimilarityScore === 'number'
        ? artifact.minSimilarityScore
        : null,
    reason: artifact.reason || null,
    sources,
  };
}

/**
 * 根据“本轮请求状态”选择要暴露给模型的工具。
 *
 * 这是 Agent 工具系统里非常关键的一层：
 *
 * 1. 工具不是越多越好
 *    如果没有图片，却把 image_analysis 暴露给模型，
 *    模型可能误判并输出“我来分析图片”。
 *
 * 2. 工具可以按上下文裁剪
 *    - hasDocs=false：不暴露 document_retrieval
 *    - hasImage=false：不暴露 image_analysis
 *    - selectedDocumentIds 有值：document_retrieval 内部强制限制检索范围
 *
 * 3. 本地任务工具常驻
 *    recipe_planner / shopping_list / nutrition_estimator / weather_lookup
 *    用来学习 Agent 如何在多个能力中选择最合适的工具。
 *
 * @param {object} params
 * @param {boolean} params.hasImage - 本轮请求是否带图片
 * @param {boolean} params.hasDocs - 当前是否有可检索文档
 * @param {string[]} params.forcedDocumentIds - 前端选中的文档范围
 * @param {Array<object>} params.scopedDocuments - 选中文档的业务信息
 * @returns {Array} 本轮允许模型调用的工具列表
 */
function selectTools({ hasImage, hasDocs, forcedDocumentIds = [], scopedDocuments = [] }) {
  const tools = [];

  // 没有文档索引时不暴露 document_retrieval，普通菜谱问题直接走 web_search。
  if (hasDocs) {
    tools.push(createDocumentRetrievalTool({
      forcedDocumentIds,
      scopedDocuments,
    }));
  }

  // 没有图片时不把 image_analysis 暴露给模型，避免模型误判后进入图片分析流程。
  if (hasImage) tools.push(imageAnalysisTool);

  // 本地任务工具常驻暴露给模型，用来学习 Agent 如何选择不同能力。
  tools.push(recipePlannerTool);
  tools.push(shoppingListTool);
  tools.push(nutritionEstimatorTool);
  tools.push(weatherLookupTool);
  tools.push(webSearchTool);
  return tools;
}

/**
 * 把可检索文档列表格式化进系统提示词。
 *
 * 模型只能看到文本提示词，不能直接读取数据库。
 * 所以这里把 documents 表里的 documentId / 文件名 / chunk 数写进 prompt，
 * 让模型在用户指定文件时能优先传 documentId 给 document_retrieval。
 *
 * @param {Array<object>} documents - documents 表中的 indexed 文档列表
 * @returns {string} 系统提示词片段
 */
function formatAvailableDocuments(documents) {
  if (!documents.length) return '- 当前没有 documents 表登记的可检索文档。';

  return documents
    .slice(0, 10)
    .map((doc) =>
      `- ${doc.originalName}（documentId: ${doc.id}，chunk: ${doc.chunkCount}）`
    )
    .join('\n');
}

/**
 * 格式化前端强制选择的文档范围。
 *
 * 当前前端可以点击侧边栏文档，限制本轮只查某些文档。
 * 这个范围会同时进入：
 * - 工具实现：强制过滤 documentId
 * - 系统提示词：提醒模型当前只允许查这些文档
 *
 * @param {Array<object>} scopedDocuments - 被前端选中的文档
 * @returns {string} 系统提示词片段
 */
function formatForcedDocumentScope(scopedDocuments) {
  if (!scopedDocuments.length) return '';

  return scopedDocuments
    .map((doc) => `- ${doc.originalName}（documentId: ${doc.id}）`)
    .join('\n');
}

/**
 * 构建本轮系统提示词。
 *
 * 系统提示词分两层：
 *
 * 1. CHEF_SYSTEM_PROMPT
 *    通用角色设定：你是私厨助手、回答风格、工具优先级等。
 *
 * 2. requestState
 *    本轮动态状态：有没有图片、有没有文档、有哪些文档、
 *    是否被前端限制检索范围、应该如何选择工具。
 *
 * 为什么每轮动态生成：
 * - Agent 不能靠“猜”当前有没有图片/文档。
 * - 如果用户没传图片，prompt 里明确禁止图片分析。
 * - 如果用户选中文档，prompt 里明确告诉模型当前检索范围。
 *
 * @param {object} params
 * @param {boolean} params.hasImage
 * @param {boolean} params.hasDocs
 * @param {Array<object>} params.documents
 * @param {Array<object>} params.scopedDocuments
 * @returns {string} 最终传给模型的 system prompt
 */
function buildSystemPrompt({
  hasImage,
  hasDocs,
  documents = [],
  scopedDocuments = [],
}) {
  const requestState = [
    '【当前请求状态】',
    hasImage
      ? '- 本轮用户已上传图片，可以在需要时调用 image_analysis。'
      : '- 本轮用户没有上传图片，禁止调用或提及图片分析，除非用户明确询问图片状态。',
    hasDocs
      ? '- 当前已有上传文档索引，可以在文档相关问题中调用 document_retrieval。'
      : '- 当前没有可检索的上传文档。普通菜谱、天气、搭配建议等问题不要说明"没有上传文档"，直接继续处理；只有用户明确询问文档内容时才说明目前没有文档。',
    hasDocs
      ? `【可检索文档】\n${formatAvailableDocuments(documents)}\n如果用户明确指定某个文件，调用 document_retrieval 时优先传入对应 documentId。`
      : null,
    scopedDocuments.length > 0
      ? `【本轮强制检索范围】\n${formatForcedDocumentScope(scopedDocuments)}\ndocument_retrieval 工具已被后端限制在上述文档内。`
      : null,
    '【工具选择规则】',
    '- 用户已经给出明确食材并要菜谱方案时，优先调用 recipe_planner。',
    '- 用户问缺什么、买什么、采购清单时，调用 shopping_list。',
    '- 用户问热量、蛋白质、减脂、营养是否均衡时，调用 nutrition_estimator。',
    '- 用户问天气、气温、下雨、是否适合外出买菜时，调用 weather_lookup。',
    '- 需要最新网络信息、图片参考或本地工具信息不足时，调用 web_search。',
    '- 不要在回答开头描述"我先检查是否有图片/文档"这类内部流程，直接给用户结果。',
  ].filter(Boolean).join('\n');

  return `${CHEF_SYSTEM_PROMPT}\n\n${requestState}`;
}

// -------------------------------------------------------
// 专家子图构建器（buildSpecialistGraph）
//
// 多 Agent 模式的核心：每个专家是一个独立的编译后子图（Sub-graph）。
//
// 子图是完整的 LangGraph 图：有自己的节点、边和 ReAct loop。
// 编译后可以作为一个节点嵌入到父图中。
//
// 父图和子图通过共享状态字段通信：
// - 输入：父图的 messages → 子图的 messages（同名字段自动传递）
// - 输出：子图的 messages → 父图的 messages（通过 reducer 合并）
//
// 子图不需要 checkpointer（由最外层父图统一管理持久化）。
//
// @param {object} params
// @param {Array} params.tools - 该专家可用的工具列表
// @param {string} params.systemPrompt - 系统提示词
// @param {string} params.name - 子图名（用于调试和日志）
// @returns {CompiledStateGraph} 编译后的子图，可作为节点嵌入父图
// -------------------------------------------------------
function buildSpecialistGraph({ tools, systemPrompt, name }) {
  // 如果该专家没有可用工具（极端情况），
  // 构建一个只有 agent 节点的最简图。
  if (tools.length === 0) {
    const callModel = async (state) => {
      const response = await model.invoke([
        new SystemMessage(systemPrompt),
        ...state.messages,
      ]);
      return { messages: response };
    };

    return new StateGraph(MessagesAnnotation)
      .addNode('agent', callModel)
      .addEdge(START, 'agent')
      .addEdge('agent', END)
      .compile({ name });
  }

  // 有工具时，构建标准 ReAct 子图：
  // agent（调用模型）→ tools（执行工具）→ agent → ... → END
  const modelWithTools = model.bindTools(tools);

  const callModel = async (state) => {
    const response = await modelWithTools.invoke([
      new SystemMessage(systemPrompt),
      ...state.messages,
    ]);
    return { messages: response };
  };

  return new StateGraph(MessagesAnnotation)
    .addNode('agent', callModel)
    .addNode('tools', new ToolNode(tools))
    .addEdge(START, 'agent')
    .addConditionalEdges('agent', toolsCondition, ['tools', END])
    .addEdge('tools', 'agent')
    .compile({ name });
}

/**
 * 构建私厨 Agent 的增强版 LangGraph。
 *
 * 增强版集成了多种 LangGraph 核心模式：
 *
 * 【图结构总览】
 *
 *   START
 *     → intent_classifier          意图分类，决定走哪条分支
 *       ├─ “off_topic”             → gentle_refuse → END
 *       │
 *       ├─ “general_chat”          → supervisor（总调度）
 *       │                             ├─ “chef_agent”     → chef_agent（烹饪专家子图）→ supervisor
 *       │                             ├─ “research_agent” → research_agent（检索专家子图）→ supervisor
 *       │                             └─ “done”           → END
 *       │
 *       └─ “meal_plan”             → meal_planner
 *                                     ├─ 有菜品 → [Send × N: dish_researcher] → meal_aggregator → END
 *                                     └─ 无菜品 → supervisor（回退到多 Agent）
 *
 * 【使用的 LangGraph 核心组件】
 *
 * 1. Sub-graph（子图嵌入）—— 多 Agent 模式
 *    - chef_agent 和 research_agent 各自是独立编译的 StateGraph
 *    - 编译后作为节点嵌入父图，父子图通过 messages 字段共享对话状态
 *    - 每个子图有自己的 ReAct loop 和工具集
 *
 * 2. Supervisor（总调度循环）—— 多 Agent 协作
 *    - supervisor 节点决定分派给哪个专家
 *    - 专家完成后回到 supervisor，supervisor 可以继续分派
 *    - 形成循环：supervisor → 专家A → supervisor → 专家B → supervisor → done
 *    - delegationCount 防止无限循环
 *
 * 3. Annotation.Root + 自定义 reducer
 *    - ChefState 扩展了 MessagesAnnotation
 *    - dishReports 使用 concat reducer，自动合并 Send 并行分支的结果
 *
 * 4. Send（动态并行扇出）
 *    - meal_planner 识别出 N 道菜后，用 Send 分发 N 个并行分支
 *    - 每个分支独立研究一道菜，彼此不阻塞
 *
 * 5. 多级条件路由（addConditionalEdges）
 *    - 第一级：intent_classifier → 按意图分三条路
 *    - 第二级：supervisor → 按分派决定选专家
 *    - 第三级：meal_planner → 按规划结果决定 Send 还是回退
 *    - 第四级：子图内部 agent → toolsCondition 决定调工具还是结束
 *
 * @param {object} params
 * @param {Array} params.tools - 本轮允许 Agent 使用的工具列表
 * @param {string} params.systemPrompt - 本轮系统提示词
 * @returns {CompiledStateGraph} 可 invoke/stream 的 LangGraph 图
 */
export function buildChefGraph({ tools, systemPrompt }) {

  // -------------------------------------------------------
  // 工具分组：把全局工具拆分给不同专家
  //
  // 为什么要拆分：
  // - 单个 Agent 面对 7 个工具时，模型可能选错工具
  // - 拆分后每个专家只看到 3-4 个工具，选择更精准
  // - 也体现了多 Agent 的分工思想：术业有专攻
  //
  // 拆分策略：
  // - 烹饪专家：recipe_planner / shopping_list / nutrition_estimator / weather_lookup
  //   都是”做菜相关”的本地规则工具
  // - 检索专家：document_retrieval / image_analysis / web_search
  //   都是”获取信息”的外部检索工具
  // -------------------------------------------------------
  const chefToolNames = new Set([
    'recipe_planner',
    'shopping_list',
    'nutrition_estimator',
    'weather_lookup',
  ]);
  const researchToolNames = new Set([
    'document_retrieval',
    'image_analysis',
    'web_search',
  ]);

  const chefTools = tools.filter((t) => chefToolNames.has(t.name));
  const researchTools = tools.filter((t) => researchToolNames.has(t.name));

  // -------------------------------------------------------
  // 构建专家子图
  //
  // 每个子图是一个独立编译的 StateGraph，
  // 编译后可以像普通节点一样 .addNode() 嵌入父图。
  //
  // 子图内部结构：agent ↔ tools（标准 ReAct loop）
  // 子图不带 checkpointer，持久化由父图统一处理。
  //
  // 两个子图共享同一个 systemPrompt，
  // 但工具集不同，模型能力自然被限制在各自领域。
  // -------------------------------------------------------
  const chefSubGraph = buildSpecialistGraph({
    tools: chefTools,
    systemPrompt,
    name: 'chef-specialist',
  });

  const researchSubGraph = buildSpecialistGraph({
    tools: researchTools,
    systemPrompt,
    name: 'research-specialist',
  });

  // -------------------------------------------------------
  // 节点一：意图分类器（intent_classifier）
  //
  // 图的入口节点。它读取用户最新消息，
  // 用模型判断意图属于 meal_plan / general_chat / off_topic。
  //
  // 返回 { intent }，只更新状态的 intent 字段，不产出 messages。
  // 这意味着分类器的模型输出不会出现在对话历史中。
  //
  // 为什么用模型而不是关键词匹配：
  // - “我冰箱里有番茄鸡蛋土豆，晚上吃啥好” → meal_plan（没有”规划”关键词）
  // - “番茄炒蛋怎么做” → general_chat（只问一道菜）
  // - 关键词很难覆盖自然语言的多样性
  // -------------------------------------------------------
  const intentClassifier = async (state) => {
    // 取用户最新消息作为分类输入。
    // 之前的对话历史不传给分类器，避免历史信息干扰当前意图判断。
    const lastMessage = state.messages[state.messages.length - 1];
    const userText = typeof lastMessage?.content === 'string'
      ? lastMessage.content
      : '';

    try {
      const response = await model.invoke([
        new SystemMessage(INTENT_CLASSIFIER_PROMPT),
        new HumanMessage(userText),
      ]);

      const raw = response.content.trim().toLowerCase();

      // 用 includes 而不是 === 是因为模型可能返回 “meal_plan。” 或 “ meal_plan “
      if (raw.includes('meal_plan')) return { intent: 'meal_plan' };
      if (raw.includes('off_topic')) return { intent: 'off_topic' };
      return { intent: 'general_chat' };
    } catch (error) {
      // 分类失败不应阻断整个请求，回退到最通用的 supervisor 分支。
      console.warn('[Graph] 意图分类失败，回退到 general_chat:', error.message);
      return { intent: 'general_chat' };
    }
  };

  // -------------------------------------------------------
  // 节点二：总调度（supervisor）
  //
  // 这是多 Agent 模式的核心协调节点。
  //
  // Supervisor 不亲自执行任务，只做三件事：
  // 1. 阅读当前对话状态
  // 2. 决定下一步交给哪个专家（或结束）
  // 3. 返回 { nextAgent, delegationCount }
  //
  // Supervisor 循环：
  //   supervisor → chef_agent → supervisor → research_agent → supervisor → done → END
  //
  // 这样可以实现多步协作：
  // 例如用户问”搜一下番茄炒蛋的做法然后帮我估算热量”
  // → supervisor 先分派 research_agent 搜索做法
  // → research_agent 完成后回到 supervisor
  // → supervisor 再分派 chef_agent 估算热量
  // → chef_agent 完成后回到 supervisor
  // → supervisor 判断任务完成，返回 done
  //
  // delegationCount 是循环保护：最多分派 3 次后强制结束，
  // 防止模型持续不返回 'done' 导致无限循环。
  // -------------------------------------------------------
  const supervisor = async (state) => {
    const count = (state.delegationCount || 0) + 1;

    // 循环保护：超过最大分派次数时强制结束。
    // 正常场景下 1-2 次分派即可完成，3 次是宽松的安全上限。
    if (count > 3) {
      console.warn('[Graph] Supervisor 达到最大分派次数，强制结束');
      return { nextAgent: 'done', delegationCount: count };
    }

    try {
      const response = await model.invoke([
        new SystemMessage(SUPERVISOR_PROMPT),
        ...state.messages,
      ]);

      const raw = response.content.trim().toLowerCase();

      let next = 'done';
      if (raw.includes('chef_agent')) next = 'chef_agent';
      else if (raw.includes('research_agent')) next = 'research_agent';

      return { nextAgent: next, delegationCount: count };
    } catch (error) {
      console.warn('[Graph] Supervisor 调用失败，强制结束:', error.message);
      return { nextAgent: 'done', delegationCount: count };
    }
  };

  // -------------------------------------------------------
  // 节点三：菜品规划器（meal_planner）
  //
  // 当意图为 meal_plan 时进入此节点。
  // 它让模型根据用户需求列出 2-4 道候选菜品，返回 JSON。
  //
  // 返回 { plannedDishes }，不产出 messages。
  // 后续 routeFromPlanner 读取 plannedDishes：
  // - 非空 → 用 Send 并行研究每道菜
  // - 为空（JSON 解析失败等）→ 回退到 supervisor 走多 Agent
  // -------------------------------------------------------
  const mealPlanner = async (state) => {
    try {
      const response = await model.invoke([
        new SystemMessage(MEAL_PLANNER_PROMPT),
        ...state.messages,
      ]);

      // 模型被要求返回严格 JSON：{ “dishes”: [“菜名1”, “菜名2”] }
      // 如果模型不遵守，JSON.parse 会抛错，catch 里回退。
      const parsed = JSON.parse(response.content);
      const dishes = Array.isArray(parsed.dishes)
        ? parsed.dishes.slice(0, 5)  // 最多 5 道菜，避免过度并行
        : [];
      return { plannedDishes: dishes };
    } catch (error) {
      // JSON 解析失败时，plannedDishes 为空，routeFromPlanner 会回退到 supervisor。
      console.warn('[Graph] 菜品规划 JSON 解析失败，将回退到 supervisor:', error.message);
      return { plannedDishes: [] };
    }
  };

  // -------------------------------------------------------
  // 节点四：菜品研究员（dish_researcher）
  //
  // 这是 Send 并行扇出的目标节点。
  //
  // Send 的工作方式：
  //   routeFromPlanner 返回 [Send('dish_researcher', state1), Send('dish_researcher', state2), ...]
  //   LangGraph 为每个 Send 创建一个独立执行分支，
  //   每个分支拿到的 state.currentDish 不同，
  //   但都指向同一个 dish_researcher 节点函数。
  //
  // 返回 { dishReports: [{ dish, report }] }。
  // 因为 dishReports 的 reducer 是 concat，
  // 多个并行分支的结果会自动合并成一个完整数组。
  //
  // 注意：这个节点不返回 messages，
  // 研究报告只存在 dishReports 里，最终由 meal_aggregator 综合输出。
  // -------------------------------------------------------
  const dishResearcher = async (state) => {
    const dish = state.currentDish || '未知菜品';

    const response = await model.invoke([
      new SystemMessage(createDishResearcherPrompt(dish)),
      new HumanMessage(`请分析这道菜：${dish}`),
    ]);

    return {
      dishReports: [{ dish, report: response.content }],
    };
  };

  // -------------------------------------------------------
  // 节点五：结果汇总器（meal_aggregator）
  //
  // 所有 dish_researcher 并行分支完成后，
  // LangGraph 通过 dishReports 的 concat reducer 把结果合并，
  // 然后流转到这个节点。
  //
  // 它读取 dishReports 数组，调用模型生成最终推荐方案。
  //
  // 返回 { messages }，这是 meal_plan 分支中唯一产出 messages 的节点，
  // 也就是唯一对用户可见的文本输出。
  // -------------------------------------------------------
  const mealAggregator = async (state) => {
    // 把每道菜的研究报告格式化成文本，交给模型综合。
    const reportsText = state.dishReports
      .map((r) => `【${r.dish}】\n${r.report}`)
      .join('\n\n');

    const response = await model.invoke([
      new SystemMessage(MEAL_AGGREGATOR_PROMPT),
      ...state.messages,
      // 把并行研究结果作为新的 HumanMessage 补充给模型，
      // 让模型能看到所有菜品的分析数据。
      new HumanMessage(`以下是各菜品的研究结果：\n\n${reportsText}\n\n请整合为最终推荐方案。`),
    ]);

    return { messages: response };
  };

  // -------------------------------------------------------
  // 节点六：礼貌拒绝（gentle_refuse）
  //
  // 当意图分类为 off_topic 时进入此节点。
  // 它不走 ReAct loop 或 Supervisor 循环，一步生成友好的引导回复。
  //
  // 这演示了条件路由的兜底分支：
  // 不是所有请求都需要复杂的工具调用或多 Agent 协作，
  // 简单场景用独立节点直达 END 即可。
  // -------------------------------------------------------
  const gentleRefuse = async (state) => {
    const response = await model.invoke([
      new SystemMessage(GENTLE_REFUSE_PROMPT),
      ...state.messages,
    ]);
    return { messages: response };
  };

  // -------------------------------------------------------
  // 路由函数一：意图路由（routeByIntent）
  //
  // 读取 intent_classifier 写入的 state.intent，
  // 返回下一个节点的名字。
  //
  // general_chat → supervisor（进入多 Agent 调度循环）
  // meal_plan → meal_planner（进入 Send 并行规划分支）
  // off_topic → gentle_refuse（直接友好拒绝）
  // -------------------------------------------------------
  function routeByIntent(state) {
    switch (state.intent) {
      case 'off_topic':
        return 'gentle_refuse';
      case 'meal_plan':
        return 'meal_planner';
      default:
        // general_chat 走 Supervisor 多 Agent 调度
        return 'supervisor';
    }
  }

  // -------------------------------------------------------
  // 路由函数二：Supervisor 分派路由（routeFromSupervisor）
  //
  // 读取 supervisor 写入的 state.nextAgent，
  // 返回下一个要执行的专家子图节点名。
  //
  // 三种可能：
  // - 'chef_agent' → 交给烹饪专家子图处理
  // - 'research_agent' → 交给检索专家子图处理
  // - 'done' / 其他 → END，结束对话
  //
  // 注意：专家完成后会回到 supervisor（通过 addEdge），
  // supervisor 再次决定是否需要继续分派，形成循环。
  // -------------------------------------------------------
  function routeFromSupervisor(state) {
    switch (state.nextAgent) {
      case 'chef_agent':
        return 'chef_agent';
      case 'research_agent':
        return 'research_agent';
      default:
        return END;
    }
  }

  // -------------------------------------------------------
  // 路由函数三：规划结果路由（routeFromPlanner）
  //
  // 这是 Send 并行扇出的核心。
  //
  // 当 meal_planner 成功识别出菜品时：
  //   返回 Send 对象数组 → LangGraph 创建 N 个并行分支
  //   每个 Send 携带不同的 currentDish
  //
  // 当 meal_planner 没有识别出菜品时（JSON 解析失败等）：
  //   返回 'supervisor' → 回退到多 Agent 调度
  //
  // Send 的参数：
  //   new Send(目标节点名, 发送给该节点的状态)
  //   每个并行分支拿到的状态是独立的副本，
  //   messages 相同（保留用户上下文），currentDish 不同。
  // -------------------------------------------------------
  function routeFromPlanner(state) {
    if (state.plannedDishes && state.plannedDishes.length > 0) {
      // 每道菜创建一个 Send，LangGraph 并行执行所有 dish_researcher。
      // 这是 Send 最典型的用法：动态 fan-out。
      return state.plannedDishes.map(
        (dish) => new Send('dish_researcher', {
          // 保留对话历史，让研究员能看到用户的原始需求
          messages: state.messages,
          // 每个并行分支处理不同的菜品
          currentDish: dish,
          // dishReports 初始化为空，每个分支独立写入
          dishReports: [],
        })
      );
    }
    // 没有识别出菜品时，回退到 supervisor 走多 Agent 调度。
    return 'supervisor';
  }

  // -------------------------------------------------------
  // 构建图
  //
  // 使用 ChefState，支持所有自定义状态字段。
  //
  // 节点类型：
  // - 普通函数节点：intentClassifier / supervisor / mealPlanner / dishResearcher / mealAggregator / gentleRefuse
  // - 子图节点：chefSubGraph / researchSubGraph（编译后的 StateGraph 直接作为节点）
  //
  // addConditionalEdges 的第三个参数是可能的目标节点列表，
  // 用于图的静态分析和 Mermaid 可视化。
  // 当路由函数返回 Send 对象时，目标节点也需要列在这里。
  // -------------------------------------------------------
  return new StateGraph(ChefState)
    // ---- 节点注册 ----

    // 入口：意图分类
    .addNode('intent_classifier', intentClassifier)

    // 多 Agent 调度：supervisor 决定分派给哪个专家
    .addNode('supervisor', supervisor)

    // 烹饪专家子图：内部有自己的 ReAct loop + 烹饪工具
    // 编译后的子图直接作为节点嵌入，LangGraph 自动处理状态映射
    .addNode('chef_agent', chefSubGraph)

    // 检索专家子图：内部有自己的 ReAct loop + 检索工具
    .addNode('research_agent', researchSubGraph)

    // 菜品规划：识别多道菜（meal_plan 分支入口）
    .addNode('meal_planner', mealPlanner)
    // 菜品研究：Send 并行目标节点
    .addNode('dish_researcher', dishResearcher)
    // 结果汇总：合并并行结果，输出最终方案
    .addNode('meal_aggregator', mealAggregator)
    // 礼貌拒绝：非烹饪话题兜底
    .addNode('gentle_refuse', gentleRefuse)

    // ---- 边（edges）----

    // 入口边：所有请求先经过意图分类
    .addEdge(START, 'intent_classifier')

    // 意图路由（第一级条件边）：
    // off_topic → gentle_refuse
    // meal_plan → meal_planner
    // general_chat → supervisor
    .addConditionalEdges('intent_classifier', routeByIntent, [
      'supervisor',
      'meal_planner',
      'gentle_refuse',
    ])

    // Supervisor 分派路由（第二级条件边）：
    // chef_agent → 烹饪专家子图
    // research_agent → 检索专家子图
    // done → END
    .addConditionalEdges('supervisor', routeFromSupervisor, [
      'chef_agent',
      'research_agent',
      END,
    ])

    // 专家完成后回到 Supervisor，形成调度循环。
    // Supervisor 会重新判断：是否需要继续分派其他专家，或者结束。
    .addEdge('chef_agent', 'supervisor')
    .addEdge('research_agent', 'supervisor')

    // 规划结果路由（第三级条件边）：
    // 有菜品 → Send 并行 dish_researcher
    // 无菜品 → 回退到 supervisor
    .addConditionalEdges('meal_planner', routeFromPlanner, [
      'dish_researcher',
      'supervisor',
    ])

    // 所有并行研究完成后，汇聚到结果汇总器
    .addEdge('dish_researcher', 'meal_aggregator')

    // 终止边
    .addEdge('meal_aggregator', END)
    .addEdge('gentle_refuse', END)

    .compile({
      checkpointer,
      name: 'chef-agent-graph',
      description: '私厨问答管家的多 Agent LangGraph（Supervisor + Sub-graph + Send 并行）',
    });
}

/**
 * 生成一个用于画图的完整 Agent 图。
 *
 * 真实聊天时，工具会按请求上下文动态裁剪：
 * - 没有图片时，不暴露 image_analysis
 * - 没有文档时，不暴露 document_retrieval
 * - 选中文档时，document_retrieval 会被限制在选中范围内
 *
 * 画图时我们希望看到“完整能力集”，所以这里直接使用 allTools。
 * 这个图只用于 getGraph()/drawMermaid()，不要拿它替代真实聊天入口。
 */
export function createDrawableChefGraph() {
  return buildChefGraph({
    tools: allTools,
    systemPrompt: CHEF_SYSTEM_PROMPT,
  });
}

/**
 * 创建“本轮请求专属”的 Agent 图和工具名集合。
 *
 * 这一步是 server.js 每次聊天都会走的入口。
 *
 * 为什么不在模块加载时只创建一个全局 graph：
 * - 每轮请求可能有不同图片状态
 * - 每轮请求可能有不同文档选择范围
 * - 文档列表会随上传/删除变化
 * - 暴露给模型的工具集也会变化
 *
 * 返回值里有两个东西：
 * - agent：已经 compile 好的 LangGraph，可 invoke/stream
 * - toolNames：本轮实际暴露的工具名，用来过滤流式 tool 事件
 *
 * @param {object} options
 * @param {boolean} options.hasImage - 本轮是否附带图片
 * @param {string[]} options.selectedDocumentIds - 前端选中的文档 id
 * @returns {Promise<{ agent: object, toolNames: Set<string> }>}
 */
async function createChefAgent(options = {}) {
  // hasImage 来自 server.js 的 imagePath。
  // 这里不检查文件本身，只决定是否把 image_analysis 工具暴露给模型。
  const hasImage = Boolean(options.hasImage);

  // hasDocuments() 会检查 pgvector 是否已有索引。
  // 如果数据库没启动，rag.js 内部会返回 false，
  // 普通聊天不应该因为向量库不可用而整体失败。
  const hasDocs = await hasDocuments();

  // selectedDocumentIds 来自前端侧边栏选中文档。
  // 空数组表示“检索全部文档”；非空表示“本轮强制只查这些文档”。
  const selectedDocumentIds = Array.isArray(options.selectedDocumentIds)
    ? options.selectedDocumentIds.filter(Boolean)
    : [];

  // documents 是给模型看的“文档目录”。
  // 模型如果要查某个文件，最好传 documentId，而不是模糊文件名。
  let documents = [];
  if (hasDocs) {
    try {
      documents = await listIndexedDocuments();
    } catch (error) {
      console.warn(`[Agent] 获取可检索文档列表失败: ${error.message}`);
    }
  }

  // 把前端传来的 selectedDocumentIds 映射成真实 indexed 文档。
  // 如果前端传了已经被删除的 id，这里会自然过滤掉。
  const selectedIdSet = new Set(selectedDocumentIds);
  const scopedDocuments = selectedDocumentIds.length > 0
    ? documents.filter((doc) => selectedIdSet.has(doc.id))
    : [];
  const forcedDocumentIds = scopedDocuments.map((doc) => doc.id);

  // 根据当前请求状态裁剪工具集。
  // 例如没有图片时，模型根本看不到 image_analysis，自然不会误调用。
  const tools = selectTools({
    hasImage,
    hasDocs,
    forcedDocumentIds,
    scopedDocuments,
  });

  // 把当前请求状态写进系统提示词。
  // 这会影响模型“是否调用工具、调用哪个工具、怎么组织回答”。
  const systemPrompt = buildSystemPrompt({
    hasImage,
    hasDocs,
    documents,
    scopedDocuments,
  });

  // 每次请求按上下文创建 LangGraph：
  // 工具集会随“是否有图片/是否有文档”变化，Agent loop 就不会走不存在的分支。
  return {
    agent: buildChefGraph({
      tools,
      systemPrompt,
    }),
    toolNames: new Set(tools.map((tool) => tool.name)),
  };
}

/**
 * 调用 Agent 处理用户消息（一次性返回）
 *
 * 这是非流式入口，目前主要用于代码测试或后续服务端同步调用。
 * 当前前端聊天使用的是 streamAgent()，因为它可以边生成边显示。
 *
 * @param {string} userMessage - 用户输入的文字
 * @param {string} threadId - 会话 ID，同一个 ID 会保留上下文记忆
 * @param {object} options - 和 streamAgent 一样的请求上下文
 * @returns {string} Agent 的最终回答
 */
export async function invokeAgent(userMessage, threadId = 'default', options = {}) {
  // configurable.thread_id 是 LangGraph checkpoint 的记忆隔离键。
  // 同一个 threadId 会接续同一段对话历史。
  const config = {
    configurable: { thread_id: threadId },
  };
  const { agent } = await createChefAgent(options);

  // invoke 会跑完整张图直到 END，然后一次性返回最终 state。
  // state.messages 里包含 HumanMessage、AIMessage、ToolMessage 等完整轨迹。
  const response = await agent.invoke(
    {
      messages: [new HumanMessage(userMessage)],
    },
    config
  );

  // 最后一条消息通常是最终 AIMessage。
  // 如果中间调用过工具，ToolMessage 已经被图循环消费过。
  const lastMessage = response.messages.at(-1);
  return lastMessage.content;
}

/**
 * 流式调用 Agent 处理用户消息
 *
 * 这是前端聊天真正使用的入口。
 *
 * LangGraph 原始 streamMode=messages 会产出 LangChain 消息对象：
 * - AIMessage：模型输出，可能是最终文本，也可能包含 tool_calls
 * - ToolMessage：工具执行结果
 *
 * 前端不应该直接处理这些 LangChain 内部对象。
 * 所以这个函数会把它们翻译成更稳定的业务事件：
 *
 * - tool_start
 *   模型决定调用某工具时发出，前端显示“正在xxx”
 *
 * - tool_result
 *   工具执行结束时发出，前端显示“xxx完成/失败”
 *
 * - rag_debug
 *   document_retrieval 的结构化调试信息，前端展示检索范围、命中数、阈值
 *
 * - sources
 *   document_retrieval 的结构化引用来源，前端渲染引用卡片
 *
 * - text
 *   最终回答的流式文本块
 *
 * @param {string} userMessage - 用户输入的文字
 * @param {string} threadId - 会话 ID
 * @param {object} options - 本轮请求上下文，例如 hasImage / selectedDocumentIds
 * @returns {AsyncGenerator} 逐块产出 Agent 事件
 */
export async function* streamAgent(userMessage, threadId = 'default', options = {}) {
  // 每个 threadId 对应一条独立对话记忆。
  // server.js 会为新会话生成 thread_xxx，切换历史会话时沿用旧 threadId。
  const config = {
    configurable: { thread_id: threadId },
  };

  // createChefAgent 会：
  // 1. 查询当前是否有文档索引
  // 2. 根据本轮是否有图片/选中文档裁剪工具
  // 3. 构建本轮系统提示词
  // 4. 编译 LangGraph
  const { agent, toolNames: activeToolNames } = await createChefAgent(options);

  // streamMode=messages 表示我们要接收“消息级”流：
  // - AIMessageChunk / AIMessage
  // - ToolMessage
  //
  // 这比只拿最终值更适合做前端 Agent loop 可视化。
  const stream = await agent.stream(
    {
      messages: [new HumanMessage(userMessage)],
    },
    {
      ...config,
      streamMode: 'messages',
    }
  );

  // startedTools：记录哪些工具步骤已经在前端显示过“正在...”
  // finishedTools：记录哪些工具步骤已经完成，避免重复完成事件
  // toolCallNames：tool_call_id -> tool name
  // toolCallArgs：tool_call_id -> tool args
  //
  // 这些 Map/Set 都只存在于一次 streamAgent 调用中。
  const startedTools = new Set();
  const finishedTools = new Set();
  const toolCallNames = new Map();
  const toolCallArgs = new Map();

  // 多 Agent 图有多个节点会调用模型，但只有部分节点的文本应该展示给用户。
  // streamMode=messages 的 metadata 里有 langgraph_node 字段，
  // 标识当前消息来自哪个节点。
  //
  // 只展示文本的节点：
  // - chef_agent：烹饪专家子图的回答（supervisor 多 Agent 路径）
  // - research_agent：检索专家子图的回答（supervisor 多 Agent 路径）
  // - agent：子图内部的 ReAct agent 节点（当 metadata 暴露子图内部节点名时）
  // - meal_aggregator：meal_plan 分支的最终推荐方案
  // - gentle_refuse：off_topic 分支的友好回复
  //
  // 不展示文本的节点（中间过程）：
  // - intent_classifier：只返回分类名，不产出 messages
  // - supervisor：只返回分派决定（chef_agent / research_agent / done），不产出 messages
  // - meal_planner：只返回 JSON 菜品列表，不产出 messages
  // - dish_researcher：只写入 dishReports，不产出 messages
  //
  // 注意：即使中间节点不返回 { messages }，
  // streamMode=messages 仍可能捕获节点内部的 LLM 调用输出。
  // 所以必须用 textVisibleNodes 过滤，防止分类结果等中间产物泄露给用户。
  //
  // 关于子图节点名的说明：
  // 当子图作为父图节点运行时，metadata.langgraph_node 可能是：
  // - 父图节点名（如 'chef_agent'）—— 子图被视为不透明节点
  // - 子图内部节点名（如 'agent'）—— 子图内部节点被展开
  // 两种都需要覆盖，所以同时列出 'agent' 和 'chef_agent' / 'research_agent'。
  const textVisibleNodes = new Set([
    'agent',            // 子图内部 ReAct agent（子图内部节点名）
    'chef_agent',       // 烹饪专家子图（父图节点名）
    'research_agent',   // 检索专家子图（父图节点名）
    'meal_aggregator',  // 结果汇总器
    'gentle_refuse',    // 礼貌拒绝
  ]);

  // LangGraph 的 stream 是 AsyncIterable。
  // 每次迭代返回 [message, metadata]：
  // - message：LangChain 消息对象（AIMessage / ToolMessage 等）
  // - metadata：包含 langgraph_node（节点名）、langgraph_step（步骤号）等
  for await (const [message, metadata] of stream) {
    const messageType = message._getType?.();
    // 当前消息来自哪个节点，用于过滤中间节点的输出
    const currentNode = metadata?.langgraph_node || '';

    // AI 消息分两类：
    // 1. tool_calls / tool_call_chunks：表示模型决定调用某个工具
    // 2. content：表示模型正在输出最终回答文本
    if (messageType === 'ai') {
      // tool_calls 是完整工具调用。
      // tool_call_chunks 是流式工具调用片段。
      // 不同模型/不同阶段可能出现其中一种，所以这里合并处理。
      const toolCalls = message.tool_calls || [];
      const toolCallChunks = (message.tool_call_chunks || []).filter((chunk) =>
        chunk.name && activeToolNames.has(chunk.name)
      );

      // 只保留本轮真实暴露给模型的工具。
      // 这样即使模型幻觉出未知工具名，前端也不会显示无效步骤。
      const toolCandidates = [...toolCalls, ...toolCallChunks].filter((toolCall) =>
        toolCall.name && activeToolNames.has(toolCall.name)
      );

      for (const toolCall of toolCandidates) {
        // key 用于把“开始事件”和“完成事件”绑定到同一个前端步骤。
        const key = getToolCallKey(toolCall);
        toolCallNames.set(key, toolCall.name);

        // args 在流式早期可能为空；如果拿到了非空 args，就缓存起来，
        // 等 tool_result 时也能把工具入参一起发给前端。
        const args = getToolCallArgs(toolCall);
        if (args) toolCallArgs.set(key, args);

        // 一个工具调用只发一次 tool_start。
        // 后续同一个工具调用的 chunk 只更新缓存，不重复渲染步骤。
        if (!startedTools.has(key)) {
          startedTools.add(key);
          const label = getToolLabel(toolCall.name);
          yield {
            type: 'tool_start',
            id: key,
            tool: toolCall.name,
            label,
            args,
            message: `正在${label}...`,
          };
        }
      }

      // 只把最终回答的文本块发给前端；工具调用消息由上面的事件单独表示。
      //
      // 三重过滤条件：
      // 1. !toolCandidates.length：排除工具调用阶段的中间文本
      // 2. textVisibleNodes.has(currentNode)：排除中间节点的输出
      //    - intent_classifier 的 "meal_plan" 分类文本不应显示
      //    - supervisor 的 "chef_agent" 分派决定不应显示
      //    - meal_planner 的 JSON 菜品列表不应显示
      //    - dish_researcher 的单菜研究报告不应显示（由 meal_aggregator 综合后输出）
      // 3. message.content 是非空字符串
      if (
        message.content &&
        typeof message.content === 'string' &&
        !toolCandidates.length &&
        textVisibleNodes.has(currentNode)
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
      // ToolMessage 会带 tool_call_id。
      // 这个 id 和前面 AIMessage.tool_calls[].id 对应。
      const key = message.tool_call_id || message.id || message.name || 'unknown_tool';
      if (finishedTools.has(key)) continue;

      finishedTools.add(key);

      // ToolMessage.name 有时不稳定，所以优先用前面缓存的 toolCallNames。
      const toolName = toolCallNames.get(key) || message.name || 'unknown_tool';
      if (!activeToolNames.has(toolName)) continue;

      const label = getToolLabel(toolName);
      const isError = message.status === 'error';

      // tool_result 是前端步骤面板的完成事件。
      // 注意这里不把完整工具结果全部塞给前端步骤，只给 preview。
      // 完整工具结果已经进入 LangGraph messages，模型能继续读取。
      yield {
        type: 'tool_result',
        id: key,
        tool: toolName,
        label,
        args: toolCallArgs.get(key),
        status: isError ? 'error' : 'success',
        message: isError ? `${label}失败` : `${label}完成`,
        preview: summarizeToolContent(message.content || ''),
      };

      // document_retrieval 会把命中的文档来源放在 ToolMessage.artifact。
      // 这部分是检索系统的事实数据，单独发给前端渲染，避免让模型手写引用时出错。
      if (
        toolName === 'document_retrieval' &&
        message.artifact?.type === 'document_sources'
      ) {
        // rag_debug 展示检索过程：query、范围、候选数、命中数、阈值。
        yield createRagDebugEvent(key, message.artifact);

        if (
          Array.isArray(message.artifact.sources) &&
          message.artifact.sources.length > 0
        ) {
          // sources 展示结构化引用来源：
          // 文件名、页码、chunkId、相关度、预览等。
          // 这比让模型在回答里手写来源更稳定。
          yield {
            type: 'sources',
            sources: message.artifact.sources,
          };
        }
      }
    }
  }
}
