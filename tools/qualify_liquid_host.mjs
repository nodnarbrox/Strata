// Actual Strata-host SSE request to the explicitly selected native Liquid worker.
import { writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
process.env.STRATA_REAL_ENGINE = '1';
process.env.STRATA_FAST_BACKEND_DISCOVERY = '1';
if (!process.env.STRATA_SERVER_URL) throw Error('Set STRATA_SERVER_URL to the dedicated native Liquid worker');
const { detectBackend, proxyCompletion } = await import('../apps/host/server/inference.mjs');
const backend = await detectBackend({ force: true });
if (backend.kind !== 'strata' || backend.modelId !== 'liquid-native') throw Error('Native Liquid worker not selected');
const rows = [];
const tasks = [
  ['What is 17 multiplied by 19? Give the final answer as just the integer.', '323'],
  ['All dax are wugs. No wugs are red. Can a dax be red? Give a final answer of just Yes or No.', 'no'],
  ['Reply with exactly the word orchid in your final answer.', 'orchid'],
];
for (const [prompt, expected] of (process.argv.includes('--quick') ? tasks.slice(0, 1) : tasks)) {
  let content = '', reasoning = '';
  const start = performance.now();
  const details = await proxyCompletion({ baseUrl: backend.baseUrl, modelId: 'liquid-native', message: prompt,
    extra: { max_tokens: 1024, temperature: 0 }, onFrame: f => {
      if (f.type === 'delta') content += f.text;
      if (f.type === 'reasoning') reasoning += f.text;
    } });
  const seconds = (performance.now() - start) / 1000;
  rows.push({ prompt, expected, content, reasoning, details, seconds,
    correct: details.finishReason === 'stop' && content.trim().toLowerCase().replace(/[.!]$/, '') === expected });
}
const tokens = rows.reduce((n, r) => n + r.details.tokensOut, 0);
const seconds = rows.reduce((n, r) => n + r.seconds, 0);
const result = { complete: true, backend, tokens, seconds, generated_tok_s: tokens / seconds,
  correct: rows.filter(r => r.correct).length, rows,
  scope: `Actual native CPU model through Strata host; ${rows.length} smoke tasks, not broad quality or reference parity` };
writeFileSync(process.argv[2], JSON.stringify(result, null, 2));
console.log(JSON.stringify({ ...result, rows: rows.map(({ prompt, reasoning, ...r }) => r) }));
if (result.correct !== rows.length || tokens <= 0 || rows.some(r => !Number.isFinite(r.details.tps) || r.details.tps > 1000)) process.exitCode = 1;
