// Natural-stop generations: no forced answer head, no ignore-EOS timing loop.
import { performance } from 'node:perf_hooks';
import { writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
const out=resolve(process.argv[2]);
process.env.STRATA_HOME=resolve(out,'strata-home');
process.env.STRATA_REAL_ENGINE='1';
process.env.STRATA_FAST_BACKEND_DISCOVERY='1';
const { detectBackend, proxyCompletion }=await import('../server/inference.mjs');
const backend=await detectBackend({force:true});
if(backend.baseUrl!=='http://127.0.0.1:1234/v1'||backend.kind!=='openai-compatible')throw Error('Unexpected endpoint');
const rows=[];
for(const [task,prompt,expected] of [
  ['arithmetic','What is 17 multiplied by 19? Give the final answer as just the integer.','323'],
  ['logic','All dax are wugs. No wugs are red. Can a dax be red? Give a final answer of just Yes or No.','no'],
  ['instruction','Reply with exactly the word orchid in your final answer.','orchid'],
]){
  let content='',reasoning='';const begin=performance.now();
  const details=await proxyCompletion({baseUrl:backend.baseUrl,modelId:backend.models[0],message:prompt,
    extra:{max_tokens:1024,temperature:0,top_k:1,repeat_penalty:1,seed:20261112,cache_prompt:false},
    onFrame:f=>{if(f.type==='delta')content+=f.text;if(f.type==='reasoning')reasoning+=f.text;}});
  const seconds=(performance.now()-begin)/1000;
  rows.push({task,prompt,expected,content,reasoning,details,seconds,
    generated_tok_s:details.tokensOut/seconds,
    natural_stop:details.finishReason==='stop',
    correct:details.finishReason==='stop'&&content.trim().toLowerCase().replace(/[.!]$/,'')===expected});
}
const tokens=rows.reduce((n,r)=>n+r.details.tokensOut,0),seconds=rows.reduce((n,r)=>n+r.seconds,0);
const report={complete:true,scope:'three simple natural-stop smoke tasks through real Strata host fallback; not heldout intelligence evidence',
  backend,actual_backend:'pinned llama.cpp b11960 CPU; candidate display name derives from tunnel port',rows,
  generated_tokens:tokens,wall_seconds:seconds,generated_tok_s:tokens/seconds,
  naturally_stopped:rows.filter(r=>r.natural_stop).length,correct:rows.filter(r=>r.correct).length};
mkdirSync(out,{recursive:true});
writeFileSync(resolve(out,'strata-complete-check.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify({...report,rows:rows.map(({content,reasoning,prompt,...r})=>r)}));
if(report.naturally_stopped!==rows.length)process.exitCode=1;
