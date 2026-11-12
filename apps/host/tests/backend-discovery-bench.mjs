// Real local HTTP endpoints; this measures discovery only, never token decode.
import { createServer } from 'node:http';
import { performance } from 'node:perf_hooks';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const home = mkdtempSync(join(tmpdir(), 'strata-discovery-bench-'));
process.env.STRATA_HOME = home;
process.env.STRATA_REAL_ENGINE = '1';
let probes = 0;
const worker = createServer((req, res) => {
  if (req.url === '/v1/status') {
    probes++;
    setTimeout(() => { res.writeHead(200, {'content-type':'application/json'}); res.end(JSON.stringify({service:'strata', model:'fixture', loaded:true})); }, 80);
  } else { res.writeHead(200, {'content-type':'application/json'}); res.end(JSON.stringify({data:[{id:'fixture'}]})); }
});
await new Promise(resolve => worker.listen(0, '127.0.0.1', resolve));
process.env.STRATA_SERVER_URL = `http://127.0.0.1:${worker.address().port}`;
try {
  const { detectBackend } = await import('../server/inference.mjs');
  await detectBackend({force:true});
  const rows = [];
  for (const force of [true, false]) {
    const start = performance.now(), before = probes;
    for (let i=0; i<10; i++) {
      const backend = await detectBackend({force});
      if (backend.kind !== 'strata' || backend.modelId !== 'fixture') throw Error('Wrong backend');
    }
    rows.push({forced:force, calls:10, worker_probes:probes-before, elapsed_ms:performance.now()-start});
  }
  console.log(JSON.stringify({scope:'real HTTP discovery, 80ms delayed worker fixture; not model generation',rows}));
} finally {
  worker.closeAllConnections();
  await new Promise(resolve=>worker.close(resolve));
  rmSync(home,{recursive:true,force:true});
}
