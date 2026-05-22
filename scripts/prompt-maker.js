// ============================================================
// prompt-maker.js —— 独立提示词优化器
//
// 作用：
//   把你写得比较简单、模糊的 prompt，优化成更详细、更可执行的 prompt。
//
// 它不会接入当前项目的 Agent，也不会影响前端和 LangGraph。
// 这是一个可以单独在终端运行的工具文件。
//
// 运行方式一：直接传参
//   npm run prompt:maker -- "优化私厨 Agent 的系统提示词" "你是一个做菜助手"
//
// 运行方式二：交互式运行
//   npm run prompt:maker
//
// 工作流程：
//   1. 从 LangChain Hub 拉取 hardkothari/prompt-maker 模板
//   2. 输入 task：你希望这个 prompt 完成什么任务
//   3. 输入 lazy_prompt：你当前写得比较粗略的 prompt
//   4. 调用模型输出优化后的 prompt
// ============================================================

import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { pull } from 'langchain/hub/node';
import { ChatOpenAI } from '@langchain/openai';
import { Client } from 'langsmith';

function printUsage() {
  console.log([
    '用法：',
    '  npm run prompt:maker -- "<你的任务>" "<你当前简短的 prompt>"',
    '  npm run prompt:maker',
    '',
    '示例：',
    '  npm run prompt:maker -- "优化私厨 Agent 的系统提示词" "你是一个做菜助手"',
  ].join('\n'));
}

async function askMultilinePrompt(rl) {
  console.log('\n请输入你当前想优化的 prompt。');
  console.log('可以输入多行；输入单独一行 END 结束。\n');

  const lines = [];
  while (true) {
    const line = await rl.question('');
    if (line.trim() === 'END') break;
    lines.push(line);
  }

  return lines.join('\n').trim();
}

async function getPromptInputs() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    printUsage();
    process.exit(0);
  }

  // 传参模式：
  // 第一个参数是 task，后面的内容都当成 lazy_prompt。
  if (args.length >= 2) {
    return {
      task: args[0],
      lazyPrompt: args.slice(1).join(' '),
    };
  }

  // 交互模式：
  // 不传参时，在终端里一步步输入。
  const rl = readline.createInterface({ input, output });
  try {
    const task = args[0] || await rl.question('你希望这个 prompt 完成什么任务？\n> ');
    const lazyPrompt = await askMultilinePrompt(rl);

    return {
      task: task.trim(),
      lazyPrompt,
    };
  } finally {
    rl.close();
  }
}

async function pullTrustedPublicPrompt(promptName) {
  const client = new Client();

  // 当前 langchain/hub/node 版本没有把 dangerouslyPullPublicPrompt
  // 从 pull() 的 options 继续传给 LangSmith Client。
  // 所以这里包一层 client，确保公共 Hub prompt 的信任确认真正传到底层。
  const trustedClient = {
    pullPromptCommit: (identifier, options = {}) =>
      client.pullPromptCommit(identifier, {
        ...options,
        dangerouslyPullPublicPrompt: true,
      }),
  };

  return pull(promptName, {
    client: trustedClient,
  });
}

const { task, lazyPrompt } = await getPromptInputs();

if (!task || !lazyPrompt) {
  console.error('缺少 task 或 lazy_prompt。');
  printUsage();
  process.exit(1);
}

if (!process.env.OPENAI_API_KEY) {
  console.error('缺少 OPENAI_API_KEY。请先在 .env 中配置模型 API Key。');
  process.exit(1);
}

// 从 LangChain Hub 拉取这个社区 Prompt。
// 官方现在把 Hub 放在 LangSmith 的 Prompts / Public Prompt Hub 里。
// 不开启 includeModel，只取 prompt 模板，再接我们自己配置的 model，风险更可控。
const prompt = await pullTrustedPublicPrompt('hardkothari/prompt-maker');

// 这里复用项目当前的 OpenAI-compatible 配置。
// 如果 .env 里 OPENAI_BASE_URL 指向 DeepSeek，也会走 DeepSeek。
const model = new ChatOpenAI({
  model: process.env.LLM_MODEL || 'deepseek-chat',
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL || 'https://api.deepseek.com/v1',
  },
  temperature: 0.2,
});

// prompt 本身是 Runnable，可以和 model pipe 成一个 chain。
// 这个 Hub prompt 的变量名来自原作者模板：
// - task：你希望 prompt 完成的任务
// - lazy_prompt：你当前写得比较粗略的 prompt
const chain = prompt.pipe(model);

console.log('\n正在优化 prompt...\n');

const response = await chain.invoke({
  task,
  lazy_prompt: lazyPrompt,
});

console.log('\n=== 优化后的 Prompt ===\n');
console.log(typeof response.content === 'string' ? response.content : response);
