"use strict";
const el = id => document.getElementById(id);
const plot = el("plot"), ctx = plot.getContext("2d");
let study, subject, repIndex = 0, playing = false, animation = 0, clockStart;
const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
const distance = (a,b) => Math.hypot(b[0]-a[0],b[1]-a[1]);
function line(c,points,color,width) {
  c.beginPath(); let first=true;
  for(const p of points) {
    if(!p || p.includes(null)){first=true;continue;}
    if(first)c.moveTo(...p);else c.lineTo(...p);first=false;
  }
  c.lineWidth=width;c.strokeStyle=color;c.lineJoin="round";c.lineCap="round";c.stroke();
}
function dot(c,p,color,r) {if(!p)return;c.beginPath();c.arc(...p,r,0,Math.PI*2);c.fillStyle=color;c.fill();}
function viewport(points,w,h,pad) {
  const xs=points.map(p=>p[0]), ys=points.map(p=>p[1]);
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const scale=Math.min((w-pad*2)/Math.max(1,xmax-xmin),(h-pad*2)/Math.max(1,ymax-ymin));
  return p=>p[0]===null||p[1]===null?null:[w/2+(p[0]-(xmin+xmax)/2)*scale,h/2+(p[1]-(ymin+ymax)/2)*scale];
}
let project;
function stop(){playing=false;cancelAnimationFrame(animation);el("play").textContent="Play reach";}
function draw(){
  const rep=subject.reps[repIndex], frames=rep.points, i=Number(el("frame").value), row=frames[i];
  ctx.fillStyle="#102637";ctx.fillRect(0,0,960,510);
  for(let k=0;k<960;k+=48)line(ctx,[[k,0],[k,510]],"#1c3545",1);
  for(let k=0;k<510;k+=48)line(ctx,[[0,k],[960,k]],"#1c3545",1);
  const trail=r=>r.map(p=>project(p.slice(4,6)));
  if(el("overlay").checked)for(const r of subject.reps)line(ctx,trail(r.points),"#587789",2);
  line(ctx,trail(frames),"#59757d",2);
  line(ctx,trail(frames.slice(0,i+1)),"#ffbc59",5);
  const arm=[0,2,4,6].map(j=>project(row.slice(j,j+2)));
  line(ctx,arm,"#b6deeb",5);arm.forEach(p=>dot(ctx,p,"#d0e2ed",6));dot(ctx,arm[2],"#ffbc59",9);
  const target=project(row.slice(8));if(target){dot(ctx,target,"#80d8b4",6);ctx.fillStyle="#b4ead4";ctx.font="18px sans-serif";ctx.fillText("Target",Math.min(890,target[0]+10),target[1]-8);}
  const speeds=rep.speeds, graph=el("speed"),pen=graph.getContext("2d");
  pen.clearRect(0,0,graph.width,graph.height);
  const peak=Math.max(1,...speeds), horizontal=j=>j*graph.width/(speeds.length-1);
  line(pen,speeds.map((v,j)=>[horizontal(j),68-v/peak*64]),"#8bc8d9",3);
  line(pen,[[horizontal(i),0],[horizontal(i),72]],"#ffbc59",3);
  el("speedValue").textContent=Math.round(speeds[i])+" mm/s";
  el("time").textContent=(i/study.fps).toFixed(2)+" s";
}
function chooseRep(index){
  stop();repIndex=index;const rep=subject.reps[index];el("frame").max=rep.points.length-1;el("frame").value=0;
  el("trial").textContent="Reach "+(index+1)+" of 8 · "+subject.side+" arm";
  const wrist=rep.points.map(p=>p.slice(4,6)), delta=distance(wrist[0],wrist.at(-1));
  rep.speeds=wrist.map((p,i)=>i?distance(p,wrist[i-1])*2*study.fps:0);
  const geom=geometry(wrist.map((p,i)=>[i,...p]));
  el("deviation").textContent="≈ "+Math.round(2*delta*geom.rel)+" mm";
  el("eff").textContent=rep.efficiency.toFixed(3);el("duration").textContent=rep.duration.toFixed(2)+" s";
  el("endpoint").textContent=rep.endpoint===null?"Unavailable":"≈ "+Math.round(rep.endpoint)+" mm";
  [...el("repButtons").children].forEach((b,j)=>b.setAttribute("aria-pressed",j===index));draw();
}
function choosePerson(){
  subject=study.cases.find(c=>c.id===el("person").value);
  const coords=subject.reps.flatMap(r=>r.points.flatMap(p=>[0,2,4,6,8].map(j=>p.slice(j,j+2)).filter(a=>!a.includes(null))));
  project=viewport(coords,960,510,42);
  el("answer").replaceChildren();el("reveal").hidden=false;chooseRep(0);
}
el("reveal").onclick=()=>{
  const model=subject.forest, label=model.call?"DCP":"TD";
  el("answer").innerHTML="<strong>Predicted "+label+" · recorded "+subject.group+"</strong><p>Random Forest · held-out fold "+model.fold+" · DCP score "+model.score.toFixed(3)+"</p>";
  const list=document.createElement("ul");
  model.terms.slice(0,3).forEach(([name,v])=>{
    const li=document.createElement("li");
    li.textContent=name.replaceAll("_"," ")+"  "+(v>=0?"+":"")+(100*v).toFixed(1)+" pp";
    list.append(li);
  });
  el("answer").append("Largest shifts in model score along the decision paths",list);
  const save=document.createElement("button");save.textContent="Download case report";
  save.onclick=()=>saveFile("participant-"+subject.id+".txt","text/plain",[
    "Reach Lab | Held-out study report","Participant "+subject.id+" | "+subject.side+" arm",
    "Random Forest | outer fold "+model.fold,"Recorded "+subject.group+" | predicted "+label+" | DCP score "+model.score.toFixed(4),
    "Training-fold baseline probability "+model.base.toFixed(4),"", "Largest decision-path contributions (percentage points)",
    ...model.terms.slice(0,5).map(([name,v])=>name+"  "+(100*v).toFixed(2)),
    "", "Repetition, path efficiency, duration (s), endpoint distance (approx. mm)",
    ...subject.reps.map((r,i)=>[i+1,r.efficiency.toFixed(3),r.duration.toFixed(2),r.endpoint?.toFixed(1)??"unavailable"].join(", ")),
    "", "Research cohort measurement. Data adapted from Haberfehlner et al., https://doi.org/10.48804/BZM9SJ (CC BY 4.0)."
  ].join("\n"));el("answer").append(save);
  el("reveal").hidden=true;
};
el("play").onclick=()=>{
  if(playing){stop();return;}
  if(el("frame").value===el("frame").max)el("frame").value=0;
  clockStart=performance.now()-Number(el("frame").value)*1000/study.fps;playing=true;el("play").textContent="Pause";
  function tick(now){const i=Math.floor((now-clockStart)*study.fps/1000);el("frame").value=i;draw();
    if(i>=Number(el("frame").max))stop();else animation=requestAnimationFrame(tick);}
  animation=requestAnimationFrame(tick);
};
el("frame").oninput=()=>{stop();draw();};el("overlay").onchange=draw;el("person").onchange=choosePerson;
const camera=el("camera"), cc=camera.getContext("2d",{willReadFrequently:true}), video=el("video");
const scratch=document.createElement("canvas");scratch.width=320;scratch.height=180;
const sc=scratch.getContext("2d",{willReadFrequently:true});
let stream=null, cameraFrame=0, color=null, pos=null, active=null, reps=[], source="Local camera",lastFrame=-1,lastKnown=null,gapStart=null,pendingCamera=0;
const say=s=>{el("status").textContent=s;};
function stopCamera(){
  pendingCamera++;cancelAnimationFrame(cameraFrame);
  if(stream)stream.getTracks().forEach(t=>t.stop());
  stream=null;video.srcObject=null;color=null;pos=null;active=null;lastFrame=-1;lastKnown=null;
  el("record").disabled=true;el("record").textContent="Start reach";el("enable").disabled=false;
}
function tab(live){
  stop();stopWatching();if(!live){stopCamera();say("Enable the camera, then click the wrist marker.");}
  el("live").hidden=!live;el("study").hidden=live;
  el("liveTab").setAttribute("aria-pressed",live);el("studyTab").setAttribute("aria-pressed",!live);
}
el("studyTab").onclick=()=>tab(false);el("liveTab").onclick=()=>tab(true);
function geometry(path){
  const p=path.map(v=>v.slice(1)), a=p[0],b=p.at(-1),d=distance(a,b);
  const len=p.slice(1).reduce((sum,v,j)=>sum+distance(v,p[j]),0);
  if(d<1||!len)throw Error("Move the marker farther and repeat the reach.");
  const dev=Math.max(...p.map(v=>Math.abs((b[0]-a[0])*(a[1]-v[1])-(a[0]-v[0])*(b[1]-a[1]))/d));
  return {eff:d/len,rel:dev/d};
}
function summary(){
  el("count").textContent=reps.length+" / 8 reaches";el("download").disabled=!reps.length;
  el("record").disabled=!stream||!color||reps.length>=8;
  el("liveStats").replaceChildren();if(!reps.length)return;
  const metrics=reps.map(geometry),eff=metrics.map(m=>m.eff),avg=mean(eff);
  const values=[avg,Math.sqrt(mean(eff.map(x=>(x-avg)**2))),mean(metrics.map(m=>m.rel))];
  const p=document.createElement("p");p.textContent=source+" · Mean efficiency "+avg.toFixed(3);el("liveStats").append(p);
  if(reps.length===8){
    const m=study.model,z=values.reduce((s,v,i)=>s+(v-m.mean[i])/m.scale[i]*m.weights[i],m.intercept);
    const text=document.createElement("p");text.textContent="Exploratory model score "+(1/(1+Math.exp(-z))).toFixed(3)+". New-device scores require validation.";el("liveStats").append(text);
  }
}
function track(data){
  // Find the nearest matching component to the previous marker position.
  const w=320,h=180,mask=new Uint8Array(w*h),work=new Int32Array(w*h);
  for(let j=0;j<mask.length;j++){
    const i=j*4;mask[j]=(data[i]-color[0])**2+(data[i+1]-color[1])**2+(data[i+2]-color[2])**2<48**2;
  }
  let best=null,cost=Infinity;
  for(let j=0;j<mask.length;j++)if(mask[j]){
    let n=1,k=0,sx=0,sy=0;work[0]=j;mask[j]=0;
    while(k<n){const v=work[k++],x=v%w,y=Math.floor(v/w);sx+=x;sy+=y;
      for(const q of [x?v-1:-1,x<w-1?v+1:-1,v-w,v+w])if(q>=0&&q<w*h&&mask[q]){mask[q]=0;work[n++]=q;}}
    if(n<6||n>2400)continue;
    const p=[sx/n,sy/n],dist=lastKnown?distance(p,lastKnown):0;
    if(dist<65&&dist<cost){best=p;cost=dist;}
  }
  return best;
}
function cameraTick(now){
  if(!stream)return;
  if(video.readyState>=2&&video.currentTime!==lastFrame){
    lastFrame=video.currentTime;
    const factor=Math.min(320/video.videoWidth,180/video.videoHeight),w=video.videoWidth*factor,h=video.videoHeight*factor;
    sc.fillStyle="#102637";sc.fillRect(0,0,320,180);sc.drawImage(video,(320-w)/2,(180-h)/2,w,h);cc.drawImage(scratch,0,0,640,360);
    if(active?.length && (now-active.at(-1)[0]>350 || active.length>=3000)){
      active=null;el("record").textContent="Start reach";say("Recording interrupted. Repeat this reach.");
    }
    if(color){
      pos=track(sc.getImageData(0,0,320,180).data);
      if(pos){lastKnown=pos;gapStart=null;dot(cc,pos.map(x=>x*2),"#ffbc59",6);if(active)active.push([Math.round(now*1000)/1000,...pos]);}
      else{gapStart??=now;if(now-gapStart>250&&active){active=null;el("record").textContent="Start reach";say("Tracking lost. Click the marker again and repeat this reach.");el("record").disabled=true;color=null;}}
      if(active && active.length && el("broadcast").checked && now-lastPacket>125 && !sending){lastPacket=now;publish("/stream",{device:"Camera wrist marker",rep:reps.length+1,path:active.slice(-1200)});}
      if(active)line(cc,active.map(p=>p.slice(1).map(v=>2*v)),"#ffbc59",3);
    }
  }
  cameraFrame=requestAnimationFrame(cameraTick);
}
el("enable").onclick=async()=>{
  stopWatching();if(source!=="Local camera"){reps=[];source="Local camera";summary();}
  const request=++pendingCamera;el("enable").disabled=true;
  try{const media=await navigator.mediaDevices.getUserMedia({video:{width:640,height:360,aspectRatio:16/9},audio:false});
    if(request!==pendingCamera){media.getTracks().forEach(t=>t.stop());return;}
    stream=media;video.srcObject=stream;await video.play();if(request!==pendingCamera)return;
    say("Click a bright, distinctly coloured wrist marker in the picture.");cameraFrame=requestAnimationFrame(cameraTick);
  }catch(e){if(request===pendingCamera){stopCamera();say("Camera unavailable. Allow camera access on localhost, or use Study replay or a device session.");}}
};
camera.onclick=e=>{
  if(!stream||active||video.readyState<2)return;
  const box=camera.getBoundingClientRect(),x=Math.min(319,Math.floor((e.clientX-box.left)*320/box.width)),y=Math.min(179,Math.floor((e.clientY-box.top)*180/box.height));
  const rgb=Array.from(sc.getImageData(x,y,1,1).data).slice(0,3);
  if(Math.max(...rgb)-Math.min(...rgb)<45){say("Select a saturated colour on the wrist marker.");return;}
  color=rgb;lastKnown=[x,y];pos=null;summary();say("Marker selected. Start at rest and finish at the target.");
};
el("record").onclick=()=>{
  if(active){const path=active;active=null;el("record").textContent="Start reach";
    if(path.length<5||distance(path[0].slice(1),path.at(-1).slice(1))<12){say("Too little tracked movement. Repeat this reach.");return;}
    reps.push(path);if(reps.length===8 && el("broadcast").checked)publish("/sessions",{device:"Camera wrist marker",reps});summary();say(reps.length===8?"Session complete. Export your measurements.":"Reach saved. Return to the starting position.");}
  else if(pos){active=[];gapStart=null;el("record").textContent="Finish reach";say("Recording wrist movement.");}
  else say("Marker missing. Click it again in the picture.");
};
el("reset").onclick=()=>{stopWatching();active=null;reps=[];source="Local camera";el("record").textContent="Start reach";el("deviceStatus").textContent="";if(!stream)cc.clearRect(0,0,640,360);summary();say(stream?"Ready for a new session.":"Enable the camera or load a device session.");};
function saveFile(name,type,body){
  const a=document.createElement("a"),url=URL.createObjectURL(new Blob([body],{type}));
  a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
el("download").onclick=()=>saveFile("reach-session.json","application/json",
  JSON.stringify({device:source,reps,measurements:reps.map(geometry),summary:el("liveStats").innerText}));
el("receive").onclick=async()=>{stopWatching();
  try{const res=await fetch("/sessions"),session=(await res.json()).at(-1);
    if(!session){el("deviceStatus").textContent="Waiting for a device to submit eight reaches.";return;}
    stopCamera();reps=session.reps;source=session.device;summary();const paths=reps.map(r=>r.map(p=>p.slice(1)));
    const transform=viewport(paths.flat(),640,360,30);cc.fillStyle="#102637";cc.fillRect(0,0,640,360);for(const path of paths)line(cc,path.map(transform),"#f6b95e",2);
    el("deviceStatus").textContent="Received eight reaches from "+source;say("Device session loaded. Reset to start a camera session.");
  }catch{el("deviceStatus").textContent="Could not load device data. Check the local server.";}
};
async function init(){
  const bundled=el("studyData");
  if(bundled){study=JSON.parse(bundled.textContent);}
  else{const res=await fetch("/study.json");if(!res.ok)throw Error();study=await res.json();}
  for(const c of study.cases){const opt=new Option("Participant "+c.id,c.id);el("person").add(opt);}
  for(let j=0;j<8;j++){const b=document.createElement("button");b.textContent=j+1;b.setAttribute("aria-label","Reach "+(j+1));b.onclick=()=>chooseRep(j);el("repButtons").append(b);}
  el("person").value="67";choosePerson();
  const [[tn,fp],[fn,tp]]=study.cm;
  const correct=study.cases.filter(c=>c.forest.call===Number(c.group==="DCP")).length;
  el("validation").textContent="Study replay Random Forest · "+correct+"/51 held-out participants correctly classified. The camera model uses three wrist-path measurements and classified "+(tn+tp)+"/51 on the study cohort.";
}
init().catch(()=>{el("trial").textContent="Study data unavailable. Run rebuild.sh and reload.";});

let lastPacket=0,sending=0,watching=null,watchVersion=0;
async function publish(path,obj){
  sending++;
  try{const res=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(obj)});
    if(!res.ok)throw Error();el("deviceStatus").textContent=path==="/sessions"?"Eight reaches sent.":"Transmitting wrist coordinates.";
  }catch{el("deviceStatus").textContent="Receiver unavailable. The local recording continues.";}finally{sending--;}
}
function stopWatching(){watchVersion++;clearInterval(watching);watching=null;el("watch").textContent="Watch device stream";}
el("watch").onclick=()=>{
  if(watching){stopWatching();return;}
  stopCamera();const version=++watchVersion;el("watch").textContent="Stop watching";say("Waiting for timestamped wrist coordinates.");
  async function refresh(){
    try{const packet=await (await fetch("/stream")).json();if(version!==watchVersion)return;
      if(!packet.path || Date.now()/1000-packet.received>3){el("deviceStatus").textContent="Waiting for fresh packets.";return;}
      const path=packet.path.map(p=>p.slice(1)),project=viewport(path,640,360,36);
      cc.fillStyle="#102637";cc.fillRect(0,0,640,360);line(cc,path.map(project),"#ffbc59",3);dot(cc,project(path.at(-1)),"#fff",5);
      el("deviceStatus").textContent=packet.device+" · reach "+packet.rep+" · "+path.length+" samples";
    }catch{if(version===watchVersion)el("deviceStatus").textContent="Connection interrupted.";}
  }
  watching=setInterval(refresh,200);refresh();
};
