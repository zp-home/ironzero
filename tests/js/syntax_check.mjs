// Syntax-check the browser ES modules with Node's own parser.
// The frontend has no bundler in this repo, so this is the closest thing to a
// compile step: it catches a stray comma or a broken import in a module the
// page loads directly.
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const here = path.dirname(fileURLToPath(import.meta.url));
const jsDir = path.resolve(here, '..', '..', 'web', 'frontend', 'js');
const entries = (await readdir(jsDir)).filter((name) => name.endsWith('.js')).sort();
let failures = 0;
for (const name of entries) {
  const full = path.join(jsDir, name);
  const source = await readFile(full, 'utf8');
  try {
    // SourceTextModule parses with module semantics (import/export) but never
    // evaluates, so no DOM or network is touched.
    new vm.SourceTextModule(source, { identifier: full });
    console.log(`OK   ${name}`);
  } catch (error) {
    failures += 1;
    console.error(`FAIL ${name}: ${error.message}`);
  }
}
console.log(`\n${entries.length - failures}/${entries.length} modules parsed`);
process.exit(failures ? 1 : 0);
