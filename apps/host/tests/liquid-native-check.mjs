// Real model inference. Requires an explicitly started llama backend/tunnel.
import { performance } from 'node:perf_hooks';
import { createHash } from 'node:crypto';
import { writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
const out = resolve(process.argv[2]);
mkdirSync(out,{recursive:true});
process.env.STRATA_HOME=resolve(out,'strata-home');
process.env.STRATA_REAL_ENGINE='1';
const { detectBackend, proxyCompletion } = await import('../server/inference.mjs');
const backend=await detectBackend({force:true});
if (backend.kind!=='openai-compatible' || backend.baseUrl!=='http://127.0.0.1:1234/v1') throw Error('Unexpected backend: '+JSON.stringify(backend));
const settings={max_tokens:128,temperature:0,top_k:1,repeat_penalty:1,seed:20261112,ignore_eos:true,cache_prompt:false};
const prompt='Explain why binary search requires sorted input. Give a small Python implementation and state its time complexity.';
const hash=text=>createHash('sha256').update(text).digest('hex');
const rows=[];
for (const arm of ['direct','strata','direct-repeat']) {
  const begin=performance.now();
  let content='',reasoning='',tokens,details;
  if (arm==='strata') {
    details=await proxyCompletion({baseUrl:backend.baseUrl,modelId:backend.models[0],message:prompt,extra:settings,
      onFrame:frame=>{if(frame.type==='delta')content+=frame.text; if(frame.type==='reasoning')reasoning+=frame.text;}});
    tokens=details.tokensOut;
  } else {
    const res=await fetch(backend.baseUrl+'/chat/completions',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({model:backend.models[0],messages:[{role:'user',content:prompt}],stream:false,...settings})});
    if(!res.ok)throw Error(await res.text());
    const raw=await res.json();
    writeFileSync(resolve(out,arm+'.raw.json'),JSON.stringify(raw,null,2));
    content=raw.choices[0].message.content??'';
    reasoning=raw.choices[0].message.reasoning_content??'';
    tokens=raw.usage.completion_tokens;details={timings:raw.timings,usage:raw.usage};
  }
  const seconds=(performance.now()-begin)/1000;
  if(!Number.isFinite(tokens)||tokens<=0)throw Error('No actual completion token count');
  rows.push({arm,generated_tokens:tokens,wall_seconds:seconds,generated_tok_s:tokens/seconds,content,reasoning,content_sha256:hash(content),reasoning_sha256:hash(reasoning),details});
}
const parity=rows.every(r=>r.content_sha256===rows[0].content_sha256&&r.reasoning_sha256===rows[0].reasoning_sha256&&r.generated_tokens===rows[0].generated_tokens);
const report={complete:true,backend,scope:'real native generation through Strata host llama.cpp fallback; no native Strata Liquid kernels',parity,rows};
writeFileSync(resolve(out,'strata-native-check.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify({...report,rows:rows.map(({content,reasoning,...r})=>r)}));
if(!parity)process.exitCode=1;
