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

function normalizeTextList(value) {
  return Array.isArray(value)
    ? value.map((item) => String(item).trim()).filter(Boolean)
    : [];
}

function formatNumber(value, digits = 0) {
  return Number(value).toFixed(digits).replace(/\.0+$/, '');
}

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
// Tavily 联网搜索客户端
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

// -------------------------------------------------------
// 工具三：菜谱规划工具
//
// 这个工具是本地规则工具，不调用模型。
// 它的价值是让 Agent 在“已经知道食材”的情况下，
// 先得到结构化候选方案，再由 LLM 组织成自然语言回答。
// -------------------------------------------------------
const recipeTemplates = [
  {
    name: '番茄炒蛋',
    required: ['番茄', '鸡蛋'],
    optional: ['葱', '蒜', '盐', '糖'],
    minutes: 12,
    difficulty: '简单',
    nutrition: '优质蛋白 + 番茄红素',
    steps: ['番茄切块，鸡蛋打散', '先炒鸡蛋盛出', '炒番茄出汁后回锅鸡蛋调味'],
  },
  {
    name: '青椒土豆丝',
    required: ['土豆', '青椒'],
    optional: ['醋', '蒜', '干辣椒'],
    minutes: 15,
    difficulty: '简单',
    nutrition: '主食替代 + 膳食纤维',
    steps: ['土豆切丝冲洗淀粉', '热锅爆香蒜末', '大火快炒并加醋保持脆感'],
  },
  {
    name: '鸡胸肉蔬菜碗',
    required: ['鸡胸肉'],
    optional: ['西兰花', '胡萝卜', '玉米', '生菜'],
    minutes: 25,
    difficulty: '中等',
    nutrition: '高蛋白 + 低脂',
    steps: ['鸡胸肉腌制后煎熟', '蔬菜焯水或煎香', '按蛋白质、蔬菜、主食分区装盘'],
  },
  {
    name: '菌菇豆腐汤',
    required: ['豆腐'],
    optional: ['香菇', '金针菇', '鸡蛋', '葱'],
    minutes: 18,
    difficulty: '简单',
    nutrition: '植物蛋白 + 清淡低负担',
    steps: ['菌菇洗净切段', '清水煮开后下豆腐和菌菇', '出锅前调味并撒葱花'],
  },
];

function scoreRecipe(template, ingredientSet) {
  const requiredHits = template.required.filter((item) => ingredientSet.has(item)).length;
  const optionalHits = template.optional.filter((item) => ingredientSet.has(item)).length;
  return requiredHits * 10 + optionalHits * 2;
}

export const recipePlannerTool = tool(
  async ({ ingredients = [], mealType = '正餐', servings = 1, maxMinutes = 30 }) => {
    const normalizedIngredients = normalizeTextList(ingredients);
    const ingredientSet = new Set(normalizedIngredients);
    const rankedRecipes = recipeTemplates
      .map((template) => ({
        ...template,
        score: scoreRecipe(template, ingredientSet),
      }))
      .filter((recipe) => recipe.score > 0 && recipe.minutes <= Number(maxMinutes || 30))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    if (rankedRecipes.length === 0) {
      return [
        `已收到食材：${normalizedIngredients.join('、') || '未提供明确食材'}。`,
        '本地菜谱库没有足够匹配的方案，建议继续调用 web_search 搜索更多菜谱。',
      ].join('\n');
    }

    const formatted = rankedRecipes
      .map((recipe, index) => [
        `${index + 1}. ${recipe.name}`,
        `   类型：${mealType}；人数：${servings}；预计 ${recipe.minutes} 分钟；难度：${recipe.difficulty}`,
        `   匹配分：${recipe.score}`,
        `   营养特点：${recipe.nutrition}`,
        `   核心步骤：${recipe.steps.join(' -> ')}`,
      ].join('\n'))
      .join('\n\n');

    return `根据现有食材规划出的候选菜谱：\n\n${formatted}`;
  },
  {
    name: 'recipe_planner',
    description:
      '当用户已经提供食材，并希望获得菜谱、晚餐搭配、做饭方案时使用。' +
      '这个工具会根据食材给出结构化候选菜谱、耗时、难度和核心步骤。',
    schema: z.object({
      ingredients: z.array(z.string()).describe('用户已有食材清单，例如 ["番茄", "鸡蛋"]'),
      mealType: z.string().optional().describe('用餐类型，例如 早餐、午餐、晚餐、便当'),
      servings: z.number().optional().describe('用餐人数'),
      maxMinutes: z.number().optional().describe('期望最大烹饪时间，单位分钟'),
    }),
  }
);

// -------------------------------------------------------
// 工具四：购物清单工具
//
// 用于把“想做的菜”和“已有食材”对比，
// 输出缺少什么、哪些是可选补充，适合真实任务型 Agent。
// -------------------------------------------------------
const dishIngredientMap = [
  {
    keywords: ['番茄炒蛋', '西红柿炒鸡蛋'],
    required: ['番茄', '鸡蛋'],
    pantry: ['盐', '糖', '食用油'],
    optional: ['葱'],
  },
  {
    keywords: ['青椒土豆丝'],
    required: ['土豆', '青椒'],
    pantry: ['盐', '醋', '食用油'],
    optional: ['蒜', '干辣椒'],
  },
  {
    keywords: ['鸡胸肉蔬菜碗', '减脂餐'],
    required: ['鸡胸肉', '西兰花'],
    pantry: ['盐', '黑胡椒', '橄榄油'],
    optional: ['玉米', '胡萝卜', '生菜'],
  },
  {
    keywords: ['菌菇豆腐汤'],
    required: ['豆腐', '菌菇'],
    pantry: ['盐', '白胡椒'],
    optional: ['葱', '鸡蛋'],
  },
];

function findDishTemplate(targetDish = '') {
  return dishIngredientMap.find((dish) =>
    dish.keywords.some((keyword) => targetDish.includes(keyword))
  );
}

export const shoppingListTool = tool(
  async ({ targetDish, availableIngredients = [], servings = 1 }) => {
    const availableSet = new Set(normalizeTextList(availableIngredients));
    const template = findDishTemplate(targetDish);

    if (!template) {
      return `暂未找到"${targetDish}"的本地配料模板。建议调用 web_search 查询标准配料后再整理购物清单。`;
    }

    const missingRequired = template.required.filter((item) => !availableSet.has(item));
    const missingPantry = template.pantry.filter((item) => !availableSet.has(item));
    const optional = template.optional.filter((item) => !availableSet.has(item));

    return [
      `目标菜品：${targetDish}`,
      `用餐人数：${servings}`,
      `必须购买：${missingRequired.length ? missingRequired.join('、') : '无'}`,
      `基础调料检查：${missingPantry.length ? missingPantry.join('、') : '基础调料已覆盖'}`,
      `可选提升：${optional.length ? optional.join('、') : '无'}`,
      '请根据用户预算和口味，把必须购买和可选提升分开展示。',
    ].join('\n');
  },
  {
    name: 'shopping_list',
    description:
      '当用户想做某道菜，并询问还需要买什么、缺什么、购物清单时使用。',
    schema: z.object({
      targetDish: z.string().describe('目标菜品名称，例如 番茄炒蛋'),
      availableIngredients: z.array(z.string()).optional().describe('用户已经有的食材或调料'),
      servings: z.number().optional().describe('用餐人数'),
    }),
  }
);

// -------------------------------------------------------
// 工具五：营养估算工具
//
// 注意：这是粗略估算，不做医疗建议。
// 目的是让 Agent 学会把“营养/热量问题”交给专门工具处理。
// -------------------------------------------------------
const nutritionTable = {
  鸡蛋: { calories: 70, protein: 6 },
  番茄: { calories: 25, protein: 1 },
  土豆: { calories: 160, protein: 4 },
  青椒: { calories: 20, protein: 1 },
  鸡胸肉: { calories: 165, protein: 31 },
  豆腐: { calories: 90, protein: 8 },
  西兰花: { calories: 35, protein: 3 },
  米饭: { calories: 230, protein: 4 },
};

export const nutritionEstimatorTool = tool(
  async ({ items = [] }) => {
    const normalizedItems = Array.isArray(items) ? items : [];
    const lines = [];
    let totalCalories = 0;
    let totalProtein = 0;

    for (const item of normalizedItems) {
      const name = String(item.name || '').trim();
      if (!name) continue;

      const unitCount = Number(item.unitCount || 1);
      const nutrition = nutritionTable[name];
      if (!nutrition) {
        lines.push(`- ${name}：暂无本地估算数据`);
        continue;
      }

      const calories = nutrition.calories * unitCount;
      const protein = nutrition.protein * unitCount;
      totalCalories += calories;
      totalProtein += protein;
      lines.push(
        `- ${name} x ${formatNumber(unitCount, 1)}：约 ${formatNumber(calories)} kcal，蛋白质 ${formatNumber(protein, 1)} g`
      );
    }

    if (!lines.length) {
      return '没有收到可估算的食材。请让用户提供食材名称和大致份量。';
    }

    return [
      '粗略营养估算如下，实际数值会受重量、品牌和烹饪方式影响：',
      ...lines,
      `合计：约 ${formatNumber(totalCalories)} kcal，蛋白质 ${formatNumber(totalProtein, 1)} g`,
      '这只是普通饮食估算，不作为医疗或减重处方。',
    ].join('\n');
  },
  {
    name: 'nutrition_estimator',
    description:
      '当用户询问热量、蛋白质、营养是否均衡、减脂餐估算时使用。' +
      '工具只做粗略饮食估算，不提供医疗建议。',
    schema: z.object({
      items: z.array(z.object({
        name: z.string().describe('食材名称，例如 鸡蛋、番茄、鸡胸肉'),
        unitCount: z.number().optional().describe('估算份数，例如 2 表示两个鸡蛋或两份'),
      })).describe('需要估算的食材列表'),
    }),
  }
);

// -------------------------------------------------------
// 工具六：天气查询工具
//
// 天气是典型“必须查最新信息”的工具场景。
// 这里复用 Tavily，把天气查询和普通 web_search 分开，
// 方便前端观察 Agent 选择了专门工具。
// -------------------------------------------------------
export const weatherLookupTool = tool(
  async ({ location, date }) => {
    const today = new Date().toISOString().slice(0, 10);
    const targetDate = date || today;
    const query = `${location} ${targetDate} 天气 气温 降雨`;

    console.log(`[Tool] 天气查询: ${query}`);

    const results = await tavilyClient.search(query, {
      maxResults: 4,
      includeImages: false,
    });

    const formatted = results.results
      .map((result, index) =>
        `[${index + 1}] ${result.title}\n来源：${result.url}\n摘要：${result.content}`
      )
      .join('\n\n');

    return `以下是"${location}"在 ${targetDate} 附近的天气查询结果：\n\n${formatted}`;
  },
  {
    name: 'weather_lookup',
    description:
      '当用户询问今天、明天、某地天气、气温、下雨、是否适合出门买菜或户外用餐时使用。',
    schema: z.object({
      location: z.string().describe('城市或地区，例如 上海、北京、杭州'),
      date: z.string().optional().describe('日期，建议使用 YYYY-MM-DD；不填则默认今天'),
    }),
  }
);

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

// -------------------------------------------------------
// 工具七：联网搜索工具（Tavily）
//
// Tavily 返回结构化搜索结果和图片 URL。
// 当本地工具、RAG 文档都不足以回答时，用它补充最新网络信息。
// -------------------------------------------------------
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
export const allTools = [
  documentRetrievalTool,
  imageAnalysisTool,
  recipePlannerTool,
  shoppingListTool,
  nutritionEstimatorTool,
  weatherLookupTool,
  webSearchTool,
];
