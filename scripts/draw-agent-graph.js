// ============================================================
// draw-agent-graph.js —— 导出当前 Agent 的 LangGraph 图
//
// 运行：
//   npm run draw:graph
//
// 输出：
//   docs/agent-graph.mmd        LangGraph 原生图 Mermaid 源码
//   docs/agent-tools-graph.mmd  展开工具节点后的学习图 Mermaid 源码
//   docs/agent-graph.html       可直接在浏览器打开的可视化页面
//
// 说明：
// - 这个脚本不会调用模型，也不会执行工具。
// - 它只创建 LangGraph 图对象，然后通过 getGraph().drawMermaid()
//   把图结构导出成 Mermaid。
// ============================================================

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createDrawableChefGraph } from '../src/agent.js';
import { allTools } from '../src/tools.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const outputDir = path.join(projectRoot, 'docs');
const mermaidPath = path.join(outputDir, 'agent-graph.mmd');
const toolsMermaidPath = path.join(outputDir, 'agent-tools-graph.mmd');
const htmlPath = path.join(outputDir, 'agent-graph.html');

// 工具名 -> 中文标签，用于展开图中标注每个工具节点。
const toolLabels = {
  document_retrieval: 'RAG 文档检索',
  image_analysis: '图片食材分析',
  recipe_planner: '菜谱规划',
  shopping_list: '购物清单',
  nutrition_estimator: '营养估算',
  weather_lookup: '天气查询',
  web_search: '联网搜索',
};

// 增强版图的节点名 -> 中文标签，用于展开图中标注每个处理节点。
const nodeLabels = {
  intent_classifier: '意图分类',
  agent: '主 Agent（ReAct）',
  tools: 'ToolNode 执行工具',
  meal_planner: '菜品规划',
  dish_researcher: '菜品研究（Send 并行）',
  meal_aggregator: '结果汇总',
  gentle_refuse: '礼貌拒绝',
};

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function escapeMermaidLabel(value) {
  return String(value).replace(/"/g, '\\"');
}

function createExpandedToolsMermaid(tools) {
  // 多 Agent 展开图：把完整的图结构都画出来，包括：
  // 1. Supervisor 多 Agent 调度循环
  // 2. 两个专家子图及其内部工具
  // 3. Send 并行扇出结构
  // 4. 各分支的中文说明

  // 按专家分组工具
  const chefToolNames = new Set(['recipe_planner', 'shopping_list', 'nutrition_estimator', 'weather_lookup']);
  const researchToolNames = new Set(['document_retrieval', 'image_analysis', 'web_search']);

  // 烹饪专家子图内部的工具展开
  const chefToolEdges = tools
    .filter((t) => chefToolNames.has(t.name))
    .map((tool) => {
      const label = toolLabels[tool.name] || tool.name;
      return `  chef_tools --> chef_${tool.name}[“${escapeMermaidLabel(tool.name)}<br/>${escapeMermaidLabel(label)}”];`;
    })
    .join('\n');

  // 检索专家子图内部的工具展开
  const researchToolEdges = tools
    .filter((t) => researchToolNames.has(t.name))
    .map((tool) => {
      const label = toolLabels[tool.name] || tool.name;
      return `  research_tools --> research_${tool.name}[“${escapeMermaidLabel(tool.name)}<br/>${escapeMermaidLabel(label)}”];`;
    })
    .join('\n');

  // Send 并行扇出的示意
  const sendExamples = ['菜品 A', '菜品 B', '菜品 N...']
    .map((dish, i) => {
      const id = `dish_${i}`;
      return [
        `  meal_planner -. “Send” .-> ${id}[“dish_researcher<br/>${escapeMermaidLabel(dish)}”];`,
        `  ${id} --> meal_aggregator;`,
      ].join('\n');
    })
    .join('\n');

  return `%%{init: {'flowchart': {'curve': 'basis'}}}%%
graph TD;
  start([START]) --> intent_classifier[“intent_classifier<br/>意图分类”];

  %% ====== 三条意图分支 ======
  intent_classifier -. “general_chat” .-> supervisor[“supervisor<br/>总调度”];
  intent_classifier -. “meal_plan” .-> meal_planner[“meal_planner<br/>菜品规划”];
  intent_classifier -. “off_topic” .-> gentle_refuse[“gentle_refuse<br/>礼貌拒绝”];

  %% ====== Supervisor 多 Agent 调度循环 ======
  supervisor -. “chef_agent” .-> chef_agent;
  supervisor -. “research_agent” .-> research_agent;
  supervisor -. “done” .-> finish([END]);

  %% 烹饪专家子图（Sub-graph）
  subgraph chef_agent[“🍳 chef_agent 烹饪专家（子图）”]
    chef_react[“agent<br/>ReAct 循环”] -. “tool_calls” .-> chef_tools[“tools<br/>ToolNode”];
${chefToolEdges}
    chef_tools --> chef_react;
    chef_react -. “完成” .-> chef_out((“ “));
  end
  chef_agent --> supervisor;

  %% 检索专家子图（Sub-graph）
  subgraph research_agent[“🔍 research_agent 检索专家（子图）”]
    research_react[“agent<br/>ReAct 循环”] -. “tool_calls” .-> research_tools[“tools<br/>ToolNode”];
${researchToolEdges}
    research_tools --> research_react;
    research_react -. “完成” .-> research_out((“ “));
  end
  research_agent --> supervisor;

  %% ====== meal_plan 分支：Send 并行扇出 ======
  meal_planner -. “无菜品（回退）” .-> supervisor;
${sendExamples}
  meal_aggregator[“meal_aggregator<br/>结果汇总”] --> finish;

  %% ====== off_topic 分支 ======
  gentle_refuse --> finish;

  %% 样式
  style intent_classifier fill:#e8f4fd,stroke:#4a90d9
  style supervisor fill:#e8eaf6,stroke:#3f51b5
  style chef_agent fill:#e8f5e9,stroke:#4caf50
  style research_agent fill:#fff8e1,stroke:#ffc107
  style meal_planner fill:#fff3e0,stroke:#ff9800
  style meal_aggregator fill:#fff3e0,stroke:#ff9800
  style gentle_refuse fill:#fce4ec,stroke:#e91e63
`;
}

function createGraphHtml({ langGraphMermaid, expandedToolsMermaid }) {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agent LangGraph</title>
  <style>
    body {
      margin: 0;
      padding: 24px;
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }

    h1 {
      margin: 0 0 8px;
      font-size: 20px;
      font-weight: 700;
    }

    p {
      margin: 0 0 20px;
      color: #667085;
      font-size: 14px;
      line-height: 1.6;
    }

    .graph-wrap {
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 20px;
      overflow: auto;
    }

    h2 {
      margin: 24px 0 12px;
      font-size: 16px;
    }
  </style>
</head>
<body>
  <h1>私厨 Agent 多 Agent LangGraph</h1>
  <p>集成了 LangGraph 核心组件：Supervisor + Sub-graph 多 Agent 调度、Annotation 自定义状态、Send 并行扇出、多级条件路由。第一张图来自 LangGraph 的 getGraph().drawMermaid()。第二张图展开子图内部结构和工具分配。</p>
  <h2>LangGraph 原生图</h2>
  <div class="graph-wrap">
    <pre class="mermaid">${escapeHtml(langGraphMermaid)}</pre>
  </div>
  <h2>工具展开图</h2>
  <div class="graph-wrap">
    <pre class="mermaid">${escapeHtml(expandedToolsMermaid)}</pre>
  </div>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({ startOnLoad: true, theme: 'default' });
  </script>
</body>
</html>
`;
}

fs.mkdirSync(outputDir, { recursive: true });

// xray=true 会尽量展开子图。当前图比较简单，核心节点就是 agent/tools。
const graph = createDrawableChefGraph();
const drawableGraph = graph.getGraph({ xray: true });
const mermaidSource = drawableGraph.drawMermaid({
  withStyles: true,
  curveStyle: 'basis',
  wrapLabelNWords: 4,
});
const expandedToolsMermaid = createExpandedToolsMermaid(allTools);

fs.writeFileSync(mermaidPath, mermaidSource, 'utf8');
fs.writeFileSync(toolsMermaidPath, expandedToolsMermaid, 'utf8');
fs.writeFileSync(
  htmlPath,
  createGraphHtml({
    langGraphMermaid: mermaidSource,
    expandedToolsMermaid,
  }),
  'utf8'
);

console.log(`Mermaid 源码已生成：${path.relative(projectRoot, mermaidPath)}`);
console.log(`工具展开图已生成：${path.relative(projectRoot, toolsMermaidPath)}`);
console.log(`HTML 预览已生成：${path.relative(projectRoot, htmlPath)}`);
