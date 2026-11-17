"""Submit real sensor output or replay recorded study wrist coordinates over HTTP."""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

parser=argparse.ArgumentParser()
parser.add_argument("server", help="For example http://localhost:8765")
parser.add_argument("--file", type=Path, help="JSON with device and eight timestamped wrist paths")
parser.add_argument("--participant", default="67")
parser.add_argument("--stream",action="store_true",help="Replay each wrist path in real time before submitting the session")
args=parser.parse_args()
if args.file:
    body=args.file.read_bytes()
else:
    study=json.loads(Path(__file__).with_name("study.json").read_text())
    case=next(c for c in study["cases"] if c["id"]==args.participant)
    reps=[[[i*40,*frame[4:6]] for i,frame in enumerate(rep["points"])] for rep in case["reps"]]
    body=json.dumps(dict(device="Recorded device replay "+case["id"],reps=reps)).encode()
if args.stream:
    obj=json.loads(body)
    for index,points in enumerate(obj["reps"],1):
        start=time.monotonic()
        for n in range(1,len(points)+1,4):
            time.sleep(max(0,(points[n-1][0]-points[0][0])/1000-(time.monotonic()-start)))
            packet=dict(device=obj["device"],rep=index,path=points[max(0,n-1200):n])
            req=Request(args.server.rstrip("/")+"/stream",data=json.dumps(packet).encode(),headers={"Content-Type":"application/json"})
            with urlopen(req,timeout=5) as response: response.read()
request=Request(args.server.rstrip("/")+"/sessions",data=body,headers={"Content-Type":"application/json"})
print(urlopen(request,timeout=10).read().decode())
