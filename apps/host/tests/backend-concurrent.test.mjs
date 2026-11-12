import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test('opt-in concurrent discovery shares probes and refuses stale publication', async t => {
  const home = mkdtempSync(join(tmpdir(), 'strata-concurrent-'));
  const names = ['STRATA_HOME', 'STRATA_REAL_ENGINE', 'STRATA_FAST_BACKEND_DISCOVERY', 'STRATA_SERVER_URL'];
  const saved = names.map(name => [name, process.env[name]]);
  const fetch = globalThis.fetch;
  process.env.STRATA_HOME = home;
  process.env.STRATA_REAL_ENGINE = '1';
  process.env.STRATA_FAST_BACKEND_DISCOVERY = '1';
  try {
    const { detectBackend, currentBackend } = await import('../server/inference.mjs');
    let release, entered, calls, held;
    function setup(port) {
      calls = [];
      const ready = new Promise(resolve => { entered = resolve; });
      held = new Promise(resolve => { release = resolve; });
      process.env.STRATA_SERVER_URL = `http://127.0.0.1:${port}`;
      globalThis.fetch = async url => {
        url = String(url);
        calls.push(url);
        if (url.endsWith('/status')) {
          if (url.startsWith(`http://127.0.0.1:${port}/`)) { entered(); await held; }
          return Response.json({ service: 'strata', model: url, loaded: true });
        }
        return Response.json({ data: [{ id: url }] });
      };
      return ready;
    }
    await t.test('one probe serves eight cold callers', async () => {
      const ready = setup(18101);
      const requests = Array.from({ length: 8 }, () => detectBackend());
      await ready;
      assert.equal(calls.length, 1);
      release();
      const values = await Promise.all(requests);
      assert.equal(calls.length, 2);
      assert.ok(values.every(value => value === values[0]));
    });
    await t.test('endpoint change cannot publish an old slow result', async () => {
      const ready = setup(18102);
      const old = detectBackend();
      await ready;
      process.env.STRATA_SERVER_URL = 'http://127.0.0.1:18103';
      const fresh = await detectBackend();
      assert.match(fresh.modelId, /18103/);
      release();
      assert.match((await old).modelId, /18102/);
      assert.strictEqual(currentBackend(), fresh);
      const count = calls.length;
      assert.strictEqual(await detectBackend(), fresh);
      assert.equal(calls.length, count);
    });
    await t.test('forced refresh bypasses pending discovery and owns the cache', async () => {
      const ready = setup(18104);
      const old = detectBackend();
      await ready;
      globalThis.fetch = async url => String(url).endsWith('/status')
        ? Response.json({ service: 'strata', model: 'forced-new', loaded: true })
        : Response.json({ data: [{ id: 'forced-new' }] });
      const fresh = await detectBackend({ force: true });
      assert.equal(fresh.modelId, 'forced-new');
      release();
      await old;
      assert.strictEqual(currentBackend(), fresh);
      assert.strictEqual(await detectBackend(), fresh);
    });
    await t.test('unavailable results are retried after a shared batch', async () => {
      process.env.STRATA_SERVER_URL = 'http://127.0.0.1:18105';
      let count = 0;
      globalThis.fetch = async () => { count++; return new Response('', { status: 404 }); };
      const values = await Promise.all(Array.from({ length: 8 }, () => detectBackend()));
      assert.ok(values.every(value => value.kind === 'unavailable'));
      assert.equal(count, 4);
      await detectBackend();
      assert.equal(count, 8);
    });
    await t.test('disabled option retains separate worker probes', async () => {
      delete process.env.STRATA_FAST_BACKEND_DISCOVERY;
      let count = 0;
      globalThis.fetch = async url => {
        if (String(url).endsWith('/status')) {
          count++;
          return Response.json({ service: 'strata', model: 'release', loaded: true });
        }
        return Response.json({ data: [{ id: 'release' }] });
      };
      const values = await Promise.all(Array.from({ length: 8 }, () => detectBackend()));
      assert.equal(count, 8);
      assert.ok(values.every(value => value.modelId === 'release'));
    });
  } finally {
    globalThis.fetch = fetch;
    for (const [name, value] of saved) {
      if (value === undefined) delete process.env[name]; else process.env[name] = value;
    }
    rmSync(home, { recursive: true, force: true });
  }
});
