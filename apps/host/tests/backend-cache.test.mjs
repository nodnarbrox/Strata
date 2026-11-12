import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test('confirmed fallback skips repeated worker probes; force and URL changes recheck', async () => {
  const home = mkdtempSync(join(tmpdir(), 'strata-discovery-'));
  const originalFetch = globalThis.fetch;
  const originalNow = Date.now;
  const oldHome = process.env.STRATA_HOME;
  const oldUrl = process.env.STRATA_SERVER_URL;
  const oldReal = process.env.STRATA_REAL_ENGINE;
  process.env.STRATA_HOME = home;
  process.env.STRATA_REAL_ENGINE = '1';
  let calls = [];
  globalThis.fetch = async url => {
    calls.push(String(url));
    if (String(url).endsWith('/status')) return new Response('', { status: 404 });
    if (String(url) === 'http://127.0.0.1:8080/v1/models') {
      return Response.json({ data: [{ id: 'liquid-test' }] });
    }
    throw new Error('Not a test endpoint');
  };
  try {
    const { detectBackend } = await import('../server/inference.mjs');
    const first = await detectBackend({ force: true });
    assert.equal(first.kind, 'openai-compatible');
    assert.equal(calls.length, 2);
    calls = [];
    const cached = await detectBackend();
    assert.deepEqual(cached, first);
    assert.deepEqual(calls, []);
    await detectBackend({ force: true });
    assert.equal(calls.length, 2);
    calls = [];
    process.env.STRATA_SERVER_URL = 'http://127.0.0.1:18001';
    await detectBackend();
    assert.equal(calls.length, 2);
    assert.equal(calls[0], 'http://127.0.0.1:18001/v1/status');
    calls = [];
    const now = Date.now();
    Date.now = () => now + 20_001;
    await detectBackend();
    assert.equal(calls.length, 2, 'expired endpoints must be discovered again');
    Date.now = originalNow;
    calls = [];
    globalThis.fetch = async url => {
      calls.push(String(url));
      if (String(url).endsWith('/status')) return Response.json({ service:'strata',model:'native',loaded:true });
      return Response.json({data:[{id:'native'}]});
    };
    assert.equal((await detectBackend({force:true})).kind, 'strata');
    calls = [];
    assert.equal((await detectBackend()).modelId, 'native');
    assert.deepEqual(calls, [], 'confirmed native workers also reuse discovery');
  } finally {
    Date.now = originalNow;
    globalThis.fetch = originalFetch;
    for (const [name, value] of [['STRATA_HOME', oldHome], ['STRATA_SERVER_URL', oldUrl], ['STRATA_REAL_ENGINE', oldReal]]) {
      if (value === undefined) delete process.env[name]; else process.env[name] = value;
    }
    rmSync(home, { recursive: true, force: true });
  }
});
