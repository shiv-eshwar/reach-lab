"""Local presentation server. All study assets ship with the project."""
import argparse
import json
import math
import threading
import webbrowser
import time
from urllib.request import Request, urlopen
from urllib.error import URLError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parent / "demo"
ASSETS = {"/": ("index.html","text/html"), "/app.js": ("app.js","text/javascript"), "/style.css": ("style.css","text/css"), "/study.json": ("study.json","application/json")}
lock = threading.Lock()
devices = {}
latest = {}
class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, mime="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            self.reply(200,b'{"app":"reach-lab","version":1}')
        elif path == "/stream":
            with lock:
                data=json.dumps(dict(latest)).encode()
            self.reply(200,data)
        elif path == "/sessions":
            with lock:
                data = json.dumps(list(devices.values())).encode()
            self.reply(200, data)
        elif path in ASSETS:
            name, mime = ASSETS[path]
            self.reply(200, (ROOT/name).read_bytes(), mime)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path not in ("/sessions","/stream") or self.headers.get_content_type() != "application/json":
            return self.send_error(400)
        try:
            size = int(self.headers.get("Content-Length","0"))
            if not 0 < size <= 1500000:
                return self.send_error(413)
            raw = self.rfile.read(size)
            obj = json.loads(raw)
            if self.server.relay:
                req = Request(self.server.relay+self.path, data=raw, headers={"Content-Type":"application/json"})
                try:
                    with urlopen(req, timeout=2) as remote:
                        return self.reply(remote.status, remote.read())
                except (URLError, TimeoutError):
                    return self.reply(502, b'{"error":"Receiver unavailable"}')
            if self.path == "/stream":
                name,rep,path=obj["device"],obj["rep"],obj["path"]
                if (not isinstance(name,str) or not 1<=len(name)<=64 or type(rep) is not int or not 1<=rep<=8
                        or not isinstance(path,list) or not 1<=len(path)<=1200):
                    raise ValueError
                previous=-1
                for t,x,y in path:
                    if not all(type(v) in (float,int) and math.isfinite(v) and abs(v)<1e10 for v in (t,x,y)) or t<=previous:
                        raise ValueError
                    previous=t
                with lock:
                    latest.update(device=name,rep=rep,path=path,received=time.time())
                return self.reply(200,b'{"ok":true}')
            name, reps = obj["device"], obj["reps"]
            if not isinstance(name,str) or not 1 <= len(name) <= 64 or not isinstance(reps,list) or len(reps)!=8:
                raise ValueError
            for points in reps:
                if not isinstance(points,list) or not 5 <= len(points) <= 3000:
                    raise ValueError
                previous = -1
                for t,x,y in points:
                    if not all(type(v) in (float,int) and math.isfinite(v) and abs(v)<1e10 for v in (t,x,y)) or t<=previous:
                        raise ValueError
                    previous=t
                if math.dist(points[0][1:],points[-1][1:])<1:
                    raise ValueError
            record=dict(id=uuid4().hex,device=name,reps=reps)
        except (ValueError, KeyError, TypeError, OverflowError):
            return self.reply(422,b'{"error":"Expected eight timestamped wrist paths"}')
        with lock:
            if len(devices)>=8:
                del devices[next(iter(devices))]
            devices[record["id"]]=record
        self.reply(201,json.dumps({"id":record["id"]}).encode())

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--host",default="127.0.0.1")
    ap.add_argument("--port",default=8765,type=int)
    ap.add_argument("--open",action="store_true")
    ap.add_argument("--relay",help="Forward camera packets to a presentation receiver URL")
    args=ap.parse_args()
    try:
        server=ThreadingHTTPServer((args.host,args.port),Handler)
    except OSError:
        if not args.open:
            raise
        url=f"http://localhost:{args.port}"
        try:
            with urlopen(url+"/health",timeout=1) as response:
                if json.load(response)=={"app":"reach-lab","version":1}:
                    webbrowser.open(url)
                    raise SystemExit(0)
        except (URLError, TimeoutError, ValueError):
            pass
        server=ThreadingHTTPServer((args.host,0),Handler)
    server.relay=(args.relay or "").rstrip("/")
    url=f"http://localhost:{server.server_port}"
    print(f"Reach Lab is ready at {url}",flush=True)
    if args.open: webbrowser.open(url)
    server.serve_forever()
