// Development only. Never imported by the packaged UI or native application.
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');

const HOST = '127.0.0.1';
const PORT = 1438;
const CLIENT_PATH = '/__argus_dev_reload.js';
const EVENTS_PATH = '/__argus_dev_events';
const CLIENT = `(() => {
  const events = new EventSource('${EVENTS_PATH}');
  events.addEventListener('reload', () => location.reload());
  addEventListener('pagehide', () => events.close(), {once:true});
})();`;
const TYPES = {
  '.html':'text/html; charset=utf-8', '.css':'text/css; charset=utf-8',
  '.js':'text/javascript; charset=utf-8', '.mjs':'text/javascript; charset=utf-8',
  '.json':'application/json', '.svg':'image/svg+xml', '.png':'image/png',
  '.ico':'image/x-icon', '.jpg':'image/jpeg', '.jpeg':'image/jpeg',
  '.webp':'image/webp', '.avif':'image/avif', '.mp4':'video/mp4',
  '.woff':'font/woff', '.woff2':'font/woff2', '.bin':'application/octet-stream',
};
function isAsset(relative) {
  const parts = relative.split(/[\\/]/);
  return !parts.some(part => part.startsWith('.') || ['tests','node_modules'].includes(part))
    && relative !== 'package.json' && Object.hasOwn(TYPES,path.extname(relative));
}
function inside(root, file) {
  const relative = path.relative(root,file);
  return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}
async function startDevServer({root=path.resolve(__dirname,'../desktop/ui'), port=PORT, debounceMs=80}={}) {
  root = await fsp.realpath(root);
  const clients = new Set();
  let timer;
  let closed = false;
  const server = http.createServer((request,response) => {
    void serve(request,response).catch(() => {
      if (!response.headersSent) response.writeHead(404);
      response.end();
    });
  });
  async function serve(request,response) {
    const authority = `${HOST}:${server.address().port}`;
    response.setHeader('Cache-Control','no-store');
    response.setHeader('X-Content-Type-Options','nosniff');
    // Refuse rebinding/cross-origin use; this server has no IPC or proxy endpoint.
    if (request.headers.host !== authority ||
        (request.headers.origin && request.headers.origin !== `http://${authority}`)) {
      response.writeHead(403); response.end(); return;
    }
    if (!['GET','HEAD'].includes(request.method)) {
      response.writeHead(405,{'Allow':'GET, HEAD'}); response.end(); return;
    }
    const rawPath = request.url.split('?')[0];
    let pathname;
    try { pathname = decodeURIComponent(rawPath); }
    catch { response.writeHead(400); response.end(); return; }
    if (pathname === EVENTS_PATH) {
      if (request.method === 'HEAD') { response.writeHead(405); response.end(); return; }
      response.writeHead(200,{'Content-Type':'text/event-stream','Connection':'keep-alive'});
      response.write(': connected\n\n');
      clients.add(response);
      request.on('close',() => clients.delete(response));
      return;
    }
    if (pathname === CLIENT_PATH) {
      response.writeHead(200,{'Content-Type':TYPES['.js']});
      response.end(request.method === 'HEAD' ? undefined : CLIENT); return;
    }
    const relative = pathname === '/' ? 'index.html' : pathname.slice(1);
    if (!pathname.startsWith('/') || !isAsset(relative)) {
      response.writeHead(404); response.end(); return;
    }
    const file = await fsp.realpath(path.resolve(root,relative));
    const stat = await fsp.stat(file);
    if (!inside(root,file) || !stat.isFile()) {
      response.writeHead(404); response.end(); return;
    }
    const type = TYPES[path.extname(file)];
    if (type === TYPES['.html']) {
      const html = await fsp.readFile(file,'utf8');
      response.writeHead(200,{'Content-Type':type});
      // External same-origin script respects the unchanged production CSP.
      const script = `<script src="${CLIENT_PATH}" defer></script>`;
      response.end(request.method === 'HEAD' ? undefined : html.replace('</body>',`${script}</body>`));
    } else {
      let start=0, end=stat.size-1, status=200;
      if (request.headers.range) {
        const match=/^bytes=(\d*)-(\d*)$/.exec(request.headers.range);
        if (match && (match[1] || match[2])) {
          start=match[1] ? Number(match[1]) : Math.max(0,stat.size-Number(match[2]));
          end=match[1] && match[2] ? Math.min(Number(match[2]),end) : end;
        }
        if (!match || (!match[1] && !match[2]) || start>end || start>=stat.size) {
          response.writeHead(416,{'Content-Range':`bytes */${stat.size}`}); response.end(); return;
        }
        status=206; response.setHeader('Content-Range',`bytes ${start}-${end}/${stat.size}`);
      }
      response.writeHead(status,{'Content-Type':type,'Content-Length':Math.max(0,end-start+1),'Accept-Ranges':'bytes'});
      if (request.method === 'HEAD' || !stat.size) response.end();
      else {
        const stream=fs.createReadStream(file,{start,end});
        stream.on('error',()=>response.destroy());
        response.on('close',()=>stream.destroy());
        stream.pipe(response);
      }
    }
  }
  await new Promise((resolve,reject) => {
    server.once('error',reject);
    server.listen(port,HOST,() => {server.off('error',reject); resolve();});
  });
  let watcher;
  try {
    watcher = fs.watch(root,{recursive:true},(_event,filename) => {
      if (filename && !isAsset(String(filename))) return;
      clearTimeout(timer);
      timer = setTimeout(() => {
        for (const client of clients) client.write('event: reload\ndata: changed\n\n');
      },debounceMs);
    });
  } catch (error) {
    server.close();
    throw error;
  }
  return {
    url:`http://${HOST}:${server.address().port}`, server,
    async close() {
      if (closed) return;
      closed = true;
      clearTimeout(timer); watcher.close();
      for (const client of clients) client.end();
      server.closeIdleConnections();
      await new Promise((resolve,reject) => server.close(error => error ? reject(error) : resolve()));
    },
  };
}
module.exports = {HOST,PORT,CLIENT_PATH,EVENTS_PATH,startDevServer};
