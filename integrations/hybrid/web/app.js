'use strict';
/* Offline-first frontend. API answers and verification flags are never recomputed here. */
const $ = id => document.getElementById(id);
const icons = {
 panel:'M4 3h16v18H4z M10 3v18',search:'M21 21l-5-5 M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0',
 compose:'M14 4H5v15h15v-9 M10 14l1-4 8-8 3 3-8 8z',clock:'M12 8v5l3 2 M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0',
 library:'M4 4v16 M9 4v16 M14 4l5 16 M2 20h20',sparkles:'M12 3l2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5z',compass:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0 M16 8l-3 5-5 3 3-5z',
 plus:'M12 5v14 M5 12h14',folder:'M3 6h7l2 2h9v12H3z',more:'M5 12h.01 M12 12h.01 M19 12h.01',info:'M12 11v6 M12 7h.01 M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0',arrowUp:'M12 19V5 M6 11l6-6 6 6',stop:'M6 6h12v12H6z',
 math:'M5 4h14 M19 20H5l8-8-8-8',orbit:'M13 3c6 0 9 10 6 15S5 22 3 16 7 3 13 3 M4 6c4-5 15 0 16 6S13 23 8 20 0 11 4 6 M12 12h.01',code:'M8 6l-6 6 6 6 M16 6l6 6-6 6 M14 3l-4 18',close:'M6 6l12 12 M18 6L6 18',trash:'M4 6h16 M9 6V3h6v3 M6 6l1 15h10l1-15 M10 10v7 M14 10v7',copy:'M8 8h12v13H8z M16 8V3H3v13h5',bookmark:'M6 3h12v18l-6-4-6 4z',check:'M5 12l4 4L19 6'
};
function icon(name) { const ns='http://www.w3.org/2000/svg'; const s=document.createElementNS(ns,'svg');s.setAttribute('viewBox','0 0 24 24');s.setAttribute('class','icon');s.setAttribute('aria-hidden','true');const p=document.createElementNS(ns,'path');p.setAttribute('d',icons[name]||icons.info);s.append(p);return s; }
function hydrateIcons(root=document) { root.querySelectorAll('[data-icon]').forEach(n=>n.replaceChildren(icon(n.dataset.icon))); }
function el(tag,text,cls) { const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);if(cls)e.className=cls;return e; }
function action(label,name,fn,cls='') { const b=el('button',undefined,cls);b.type='button';if(name)b.append(icon(name));b.append(el('span',label));b.onclick=fn;return b; }
const STORE='mindscape.workspace.v2';
let storageAvailable=true,storageWritable=true;
function validChat(c){return c&&typeof c.id==='string'&&typeof c.title==='string'&&Array.isArray(c.messages)&&c.messages.every(m=>m&&['user','assistant'].includes(m.role)&&typeof m.content==='string');}
function readStore() {
 try {
  const raw=localStorage.getItem(STORE);let value;try{value=JSON.parse(raw||'null');}catch{value=null;}
  if(value?.version===2&&Array.isArray(value.chats)&&value.chats.every(validChat)&&Array.isArray(value.projects)&&value.projects.every(p=>p&&typeof p.id==='string'&&typeof p.name==='string')){
   value.chats.forEach(c=>c.messages.forEach(m=>{if(!m.id)m.id=crypto.randomUUID();}));
   value.saved=Array.isArray(value.saved)?value.saved.filter(x=>typeof x==='string'):[];
   value.deleted=Array.isArray(value.deleted)?value.deleted.filter(x=>validChat(x?.chat)).map(x=>({...x,saved:Array.isArray(x.saved)?x.saved:[]})):[];
   return value;
  }
  if(raw){try{localStorage.setItem(STORE+'.recovery',raw);}catch{storageWritable=false;storageAvailable=false;}}
 } catch { storageAvailable=false;storageWritable=false; }
 let legacy=[];
 try { const value=JSON.parse(localStorage.getItem('mind1.chats')||'[]');if(Array.isArray(value))legacy=value.filter(c=>typeof c.title==='string'&&Array.isArray(c.messages)).map(c=>({...c,id:c.id||crypto.randomUUID(),messages:c.messages.filter(m=>['user','assistant'].includes(m.role)&&typeof m.content==='string').map(m=>({...m,id:m.id||crypto.randomUUID(),state:'sent'}))})); }
 catch { /* Preserve malformed legacy data rather than deleting it. */ }
 return {version:2,chats:legacy,projects:['Research','CBSE Prep','Personal'].map(name=>({id:crypto.randomUUID(),name})),saved:[],collapsed:false};
}
const state=readStore();state.saved=Array.isArray(state.saved)?state.saved:[];state.deleted=Array.isArray(state.deleted)?state.deleted:[];
let current=state.chats.some(c=>c.id===state.current)?state.current:null,project=state.chats.find(c=>c.id===state.current)?.projectId||null,page='chat',active=null,lastFinished=0,modalReturn=null,traceReturn=null,toastTimer;
for(const c of state.chats)for(const m of c.messages)if(m.state==='pending')m.state='interrupted';
const drafts=new Map();
const prompts={math:'Differentiate exp(sin(2*x))*cos(x^3) with respect to x',physics:'Lens equation, f=20 cm and u=-30 cm',coding:'Predict the output:\n```python\ns=0\nfor i in range(5):\n s+=i\nprint(s)\n```'};
function persist() {
 state.current=current;
 try { if(!storageWritable)throw new Error('Protected unreadable storage');localStorage.setItem(STORE,JSON.stringify(state));storageAvailable=true; }
 catch { storageAvailable=false;showError('Browser storage is unavailable or full. This session still works; export important conversations.'); }
 $('storageNotice').textContent=storageAvailable?' Chats stay in this browser.':' Changes are not being saved.';
}
function toast(text) { clearTimeout(toastTimer);$('toast').textContent=text;$('toast').hidden=false;toastTimer=setTimeout(()=>$('toast').hidden=true,3000); }
function announce(text) { $('announcer').textContent=text; }
function showError(text,retry) { $('error').replaceChildren(el('span',text));if(retry)$('error').append(action('Retry',null,retry));$('error').hidden=false; }
function closeDialog() { $('modal').close();modalReturn?.focus(); }
function openDialog(title,build) {
 modalReturn=document.activeElement;$('modalTitle').textContent=title;$('modalBody').replaceChildren();$('modalActions').replaceChildren();$('modalForm').onsubmit=e=>e.preventDefault();build($('modalBody'),$('modalActions'));$('modal').showModal();$('modalBody').querySelector('input,textarea')?.focus();
}
function info(title,text) { openDialog(title,(body,actions)=>{body.append(el('p',text,'dialog-copy'));actions.append(action('Got it',null,closeDialog,'primary'));}); }
function projectName() { return state.projects.find(p=>p.id===project)?.name||null; }
function currentChat() { return state.chats.find(c=>c.id===current); }
function localDate() { return new Date().toISOString(); }
function conversationTitle(text){const code=text.match(/```(?:python|py)?\s*\n([^\n]+)/i);return (code?'Python · '+code[1].trim():text.trim().split('\n')[0].replace(/^#+\s*/,'')).slice(0,60);}
function setPage(next) {
 page=next;$('inspector').hidden=true;$('chatView').hidden=next!=='chat';$('collectionView').hidden=!['library','assistants','explore'].includes(next);$('researchView').hidden=next!=='research';
 document.querySelectorAll('.primary-nav .nav-item').forEach(b=>b.classList.remove('selected'));const id={chat:'newChat',library:'libraryPage',assistants:'assistantsPage',explore:'explorePage'}[next];if(id)$(id).classList.add('selected');
 if(['library','assistants','explore'].includes(next))renderCollection(next);
 closeDrawer();
}
function newChat() { if(page==='chat'&&current===null){$('prompt').value='';$('error').hidden=true;render();$('prompt').focus();return;}if(current)drafts.set(current,$('prompt').value);current=null;$('prompt').value='';$('error').hidden=true;setPage('chat');render();persist();$('prompt').focus(); }
function selectChat(id) { if(current)drafts.set(current,$('prompt').value);const c=state.chats.find(c=>c.id===id);if(!c)return;current=id;project=c.projectId||null;$('error').hidden=true;setPage('chat');$('prompt').value=drafts.get(id)||'';render();persist();$('prompt').focus(); }
function renderSidebar() {
 const projects=$('projects');projects.replaceChildren();for(const p of state.projects){const b=action(p.name,'folder',()=>{project=p.id;current=null;$('prompt').value='';setPage('chat');render();$('prompt').focus();},'nav-item project-item'+(project===p.id?' selected':''));b.title=p.name;b.querySelector('span').classList.add('sidebar-label');projects.append(b);}
 $('projectEmpty').hidden=state.projects.length>0;
 const history=$('history');history.replaceChildren();const chats=state.chats.filter(c=>!project||c.projectId===project);
 for(const c of chats){const row=el('div',undefined,'history-row'+(current===c.id?' selected':''));const b=el('button',c.title,'history-item');b.title=c.title;b.setAttribute('aria-current',current===c.id?'page':'false');b.onclick=()=>selectChat(c.id);const d=el('button',undefined,'icon-button delete-chat');d.append(icon('trash'));d.title='Delete '+c.title;d.setAttribute('aria-label','Delete conversation '+c.title);d.onclick=()=>confirmDelete(c);row.append(b,d);history.append(row);}
 $('historyEmpty').hidden=chats.length>0;$('projectContext').hidden=!project;$('projectContext').textContent=projectName()||'';$('projectContext').title='Show all projects';$('welcomeProject').hidden=!project;$('welcomeProject').textContent=projectName()||'';
}
function confirmDelete(chat) {
 if(active?.chatId===chat.id)return toast('Wait for this request to finish or stop it before deleting.');
 openDialog('Delete conversation?',(body,actions)=>{body.append(el('p','“'+chat.title+'” will move to Recently deleted in your Library. You can restore it there.','dialog-copy'));actions.append(action('Cancel',null,closeDialog),action('Delete',null,()=>{state.chats=state.chats.filter(c=>c.id!==chat.id);const ids=new Set(chat.messages.map(m=>m.id));state.deleted.push({chat,saved:state.saved.filter(id=>ids.has(id))});state.saved=state.saved.filter(id=>!ids.has(id));if(current===chat.id){current=null;$('prompt').value='';}persist();closeDialog();render();toast('Moved to Recently deleted in Library.');},'danger'));});
}
function inline(parent,text) {
 const re=/(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\(https?:\/\/[^\s)]+\))/g;let last=0;
 for(const m of text.matchAll(re)){parent.append(document.createTextNode(text.slice(last,m.index)));const t=m[0];if(t[0]==='`')parent.append(el('code',t.slice(1,-1)));else if(t.startsWith('**'))parent.append(el('strong',t.slice(2,-2)));else {const a=el('a',t.slice(1,t.indexOf(']')));a.href=t.slice(t.indexOf('(')+1,-1);a.target='_blank';a.rel='noopener noreferrer';parent.append(a);}last=m.index+t.length;}parent.append(document.createTextNode(text.slice(last)));
}
function highlighted(code,language) {
 const root=el('code');if(!/^(python|py|sql|json|javascript|js)$/.test(language)){root.textContent=code;return root;}
 const regex=/(#[^\n]*|--[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:def|return|if|else|elif|for|in|while|try|except|finally|raise|import|from|class|True|False|None|and|or|not|SELECT|FROM|WHERE|JOIN|ON|GROUP|BY|ORDER|HAVING|AS|SUM|AVG|COUNT|LIMIT|const|let|function)\b|\b\d+(?:\.\d+)?\b)/g;let offset=0;
 for(const m of code.matchAll(regex)){root.append(document.createTextNode(code.slice(offset,m.index)));const v=m[0];root.append(el('span',v,v.startsWith('#')||v.startsWith('--')?'token-comment':/^['"]/.test(v)?'token-string':/^\d/.test(v)?'token-number':'token-keyword'));offset=m.index+v.length;}root.append(document.createTextNode(code.slice(offset)));return root;
}
function markdown(text,literalProse=false) {
 const root=el('div',undefined,'message-body');let code=false,lines=[],lang='';
 function flush() { if(!lines.length)return;const content=lines.join('\n');lines=[];
  if(code){const block=el('div',undefined,'code-block');const head=el('div',undefined,'code-header');head.append(el('span',lang||'text'),action('Copy code','copy',e=>copyText(content,e.currentTarget)));const pre=el('pre');pre.append(highlighted(content,lang.toLowerCase()));block.append(head,pre);root.append(block);return;}
  if(literalProse){root.append(el('p',content));return;}
  for(const block of content.split(/\n\s*\n/)){if(window.renderTable(root,block))continue;const ls=block.split('\n');if(ls.every(l=>/^\s*(?:[-*]|\d+\.)\s+/.test(l))){const list=el(/^\s*\d+\./.test(ls[0])?'ol':'ul');for(const l of ls){const li=el('li');inline(li,l.replace(/^\s*(?:[-*]|\d+\.)\s+/,''));list.append(li);}root.append(list);continue;}const h=block.match(/^(#{1,4})\s+([\s\S]*)$/);const p=el(h?'h'+Math.min(h[1].length+1,5):'p');inline(p,h?h[2]:block);root.append(p);}
 }
 for(const line of String(text??'').split('\n')){if(/^```/.test(line)){flush();code=!code;lang=line.slice(3).trim();}else lines.push(line);}flush();if(!literalProse)window.renderMind1Math(root);return root;
}
async function copyText(text,button) {
 try { await navigator.clipboard.writeText(text);toast('Copied to clipboard.'); }
 catch { info('Copy this text',text); }
}
function provenance(result) {
 if(!result)return 'Imported answer · provenance unavailable';if(result.provenance?.native_neural_only)return 'Native research interface · unsupported chat input';
 if(result.status==='unsupported')return 'Unsupported · no verified answer';if(result.status==='cancelled')return 'Cancelled · no verified answer';
 if(result.provenance?.tool_verified)return result.provenance.pretrained_used?'Hybrid · verified solver result':'Tools only · verified solver result';
 if(result.status==='unverified')return 'Pretrained inference · unverified';return result.status||'Response';
}
function structuredTable(answer){
 if(!answer||!Array.isArray(answer.columns)||!Array.isArray(answer.rows)||answer.columns.length>20||answer.rows.length>500||!answer.rows.every(r=>Array.isArray(r)&&r.length===answer.columns.length))return null;
 const wrap=el('div',undefined,'table-wrap');const table=el('table');const head=el('thead');const heading=el('tr');
 for(const name of answer.columns){const th=el('th',name);th.scope='col';heading.append(th);}head.append(heading);table.append(head);const body=el('tbody');
 for(const row of answer.rows){const tr=el('tr');for(const value of row)tr.append(el('td',value===null?'null':Object.is(value,-0)?'-0':String(value)));body.append(tr);}table.append(body);wrap.append(table);return wrap;
}
function renderMessage(message) {
 const article=el('article',undefined,'message '+message.role);article.dataset.messageId=message.id;
 if(message.role==='assistant'){const head=el('div',undefined,'message-head');head.append(el('span','m','brand-mark'),el('span','Mindscape'));article.append(head);}
 let displayed=message.content;
 if(message.role==='assistant'&&message.result?.status==='verified'&&message.result.provenance?.tool_verified){
  const start=displayed.indexOf('\n\nAnswer: ');if(start>=0)displayed=displayed.slice(start+2);
 }
 if(message.role==='assistant'&&message.result?.provenance?.tool_verified&&message.result.structured_problem?.task==='trace'&&typeof message.result.solver_result?.answer==='string'){
  const body=el('div',undefined,'message-body');const block=el('div',undefined,'code-block');const header=el('div',undefined,'code-header');const output=message.result.solver_result.answer;
  header.append(el('span','Python output · exact text'),action('Copy output','copy',e=>copyText(output,e.currentTarget)));const pre=el('pre');pre.append(el('code',output));block.append(header,pre);body.append(block);
  const start=message.content.lastIndexOf('\n\nWorked solution:\n');if(start>=0){const details=el('details',undefined,'program-details');details.append(el('summary','Worked solution & execution trace'),el('pre',message.content.slice(start+2),'program-working'));body.append(details);}article.append(body);
 }else article.append(markdown(displayed,message.role==='user'||!!message.result?.provenance?.tool_verified));
 if(message.role==='assistant'){
  if(message.result?.provenance?.tool_verified && message.result.domain==='math' && typeof message.result.solver_result?.answer==='string'){
   const visual=window.renderMindscapeExpression(message.result.solver_result.answer);
   if(visual){const summary=el('div',undefined,'answer-summary');summary.append(el('small','Verified result · original expression retained above'),visual);article.append(summary);}
  }
  if(message.result?.provenance?.tool_verified&&message.result.domain==='sql'){const table=structuredTable(message.result.solver_result?.answer);if(table)article.append(table);}
  const meta=el('div',undefined,'provenance');meta.append(icon(message.result?.provenance?.tool_verified?'check':'info'),el('span',provenance(message.result)));article.append(meta);
  const actions=el('div',undefined,'message-actions');actions.append(action('Copy','copy',e=>copyText(message.content,e.currentTarget)),action('Activity','info',e=>openTrace(message.result,e.currentTarget)),action(state.saved.includes(message.id)?'Saved':'Save','bookmark',()=>{state.saved=state.saved.includes(message.id)?state.saved.filter(id=>id!==message.id):[...state.saved,message.id];persist();render();toast(state.saved.includes(message.id)?'Saved to your local Library.':'Removed from Library.');}));article.append(actions);
 }else if(['failed','stopped','interrupted'].includes(message.state)){const note=el('div',message.state==='stopped'?'Request stopped':'Request not completed','sr-only');article.append(note);}
 return article;
}
function render() {
 const chat=currentChat();const empty=!chat?.messages.length;$('chatView').classList.toggle('empty',empty);$('welcome').hidden=!empty;
 $('messages').replaceChildren();if(chat)for(const m of chat.messages)$('messages').append(renderMessage(m));
 if(active?.chatId===current){const pending=el('div',undefined,'message pending');pending.setAttribute('role','status');pending.append(el('span',undefined,'loading-dot'),el('span',active.stopping?'Requesting stop…':'Mindscape is working…'));$('messages').append(pending);}
 renderSidebar();resizePrompt();syncComposer();if(!empty)$('messages').scrollTop=$('messages').scrollHeight;
}
function resizePrompt() { $('prompt').style.height='auto';$('prompt').style.height=Math.min($('prompt').scrollHeight,240)+'px'; }
function syncComposer() {
 $('send').hidden=!!active;$('cancel').hidden=!active;$('cancel').disabled=!!active?.stopping;$('mode').disabled=!!active;$('send').disabled=!!active||!$('prompt').value.trim()||Date.now()-lastFinished<250;
 $('prompt').setAttribute('aria-busy',String(!!active));$('composerContext').textContent=active?'Local inference in progress':projectName()||'Ask, explore, understand';
}
function recoveryDraft(original,draft){return draft.trim()?draft:original;}
async function send(text,retryId) {
 if(active||!text.trim()||Date.now()-lastFinished<250)return;
 const mode=$('mode').value;const retainedDraft=retryId&&$('prompt').value!==text?$('prompt').value:'';let chat=currentChat();if(!chat){chat={id:crypto.randomUUID(),title:conversationTitle(text),messages:[],projectId:project,createdAt:localDate()};state.chats.unshift(chat);current=chat.id;}
 let user=chat.messages.find(m=>m.id===retryId);if(!user){const last=chat.messages.at(-1);if(last?.role==='user'&&['failed','stopped','interrupted'].includes(last.state)&&last.content===text)user=last;}
 const history=chat.messages.filter(m=>m!==user).map(({role,content})=>({role,content}));if(!user){user={id:crypto.randomUUID(),role:'user',content:text,state:'pending'};chat.messages.push(user);}else user.state='pending';
 chat.updatedAt=localDate();state.chats=[chat,...state.chats.filter(c=>c.id!==chat.id)];
 const request={id:crypto.randomUUID(),chatId:chat.id,controller:new AbortController(),stopping:false};active=request;$('error').hidden=true;$('prompt').value=retainedDraft;if(retainedDraft)drafts.set(chat.id,retainedDraft);else drafts.delete(chat.id);persist();render();announce('Request sent. Mindscape is working.');
 const timeout=setTimeout(()=>{request.timedOut=true;request.controller.abort();requestServerStop(request);},145000);
 try {
  const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,history,request_id:request.id,inference_mode:mode,native_demo:mode==='native'}),signal:request.controller.signal});
  let data;try{data=await response.json();}catch{throw new Error('The server returned an unreadable response. Your message is preserved.');}
  if(!response.ok){const detail=response.status===429?'The local model is still busy. Wait briefly, then retry.':data.error||`Request failed (${response.status}).`;throw new Error(detail);}
  if(typeof data.answer!=='string')throw new Error('The response did not contain an answer.');
  user.state='sent';chat.messages.push({id:crypto.randomUUID(),role:'assistant',content:data.answer,result:data,createdAt:localDate()});announce(data.status==='verified'?'Verified result received.':'Response received. Check its provenance.');if(current!==chat.id)toast('Response ready in “'+chat.title+'”.');
 } catch(error) {
  user.state=request.stopping?'stopped':'failed';const note=request.timedOut?'The request took too long. A stop was requested; the model may finish its in-flight call.':error.name==='AbortError'?'Stopped waiting. A stop was requested; the model may finish its in-flight call.':error.message;
  if(current===chat.id){const restored=recoveryDraft(text,$('prompt').value);$('prompt').value=restored;drafts.set(chat.id,restored);showError(note,()=>send(text,user.id));}else toast('Request incomplete in “'+chat.title+'”.');announce(note);
 } finally {
  clearTimeout(timeout);if(active===request)active=null;lastFinished=Date.now();persist();render();setTimeout(syncComposer,260);if(current===chat.id&&page==='chat')$('prompt').focus();
 }
}
async function requestServerStop(request) {
 try { const r=await fetch('/api/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({request_id:request.id}),keepalive:true});if(!r.ok)toast('Stop could not be confirmed. The server may finish the current request.'); }
 catch { toast('Connection stopped; server cancellation could not be confirmed.'); }
}
function stop() { if(!active)return;active.stopping=true;const request=active;requestServerStop(request);request.controller.abort();syncComposer(); }
function openTrace(result,button) {
 traceReturn=button;const root=$('traceContent');root.replaceChildren();
 if(!result)root.append(el('p','No execution metadata was stored for this imported answer.'));
 else for(const [label,value] of [['Status',result.status],['Method',result.method],['Interpreted problem',result.structured_problem],['Returned execution events',result.trace],['Solver result',result.solver_result],['Neural plan — advisory, unverified',result.neural],['Provenance',result.provenance],['Timing',result.timing],['Limitations',result.limitations]]){if(value===undefined)continue;const section=el('section',undefined,'trace-event');section.append(el('b',label),el('pre',typeof value==='string'?value:JSON.stringify(value,null,2)));root.append(section);}
 $('inspector').hidden=false;$('closeTrace').focus();
}
function closeTrace() { $('inspector').hidden=true;traceReturn?.focus(); }
function closeDrawer() { document.body.classList.remove('drawer-open');$('drawerBackdrop').hidden=true;$('menu').setAttribute('aria-expanded','false');$('workspace').inert=false;const mobile=matchMedia('(max-width: 800px)').matches;$('sidebar').inert=mobile;const label=mobile?'Close sidebar':state.collapsed?'Expand sidebar':'Collapse sidebar';$('collapseSidebar').setAttribute('aria-label',label);$('collapseSidebar').title=label; }
function openDrawer() { document.body.classList.add('drawer-open');$('drawerBackdrop').hidden=false;$('menu').setAttribute('aria-expanded','true');$('workspace').inert=true;$('sidebar').inert=false;$('collapseSidebar').focus(); }
function collapseSidebar() {
 if(matchMedia('(max-width: 800px)').matches){closeDrawer();$('menu').focus();return;}
 state.collapsed=!state.collapsed;document.body.classList.toggle('sidebar-collapsed',state.collapsed);$('collapseSidebar').setAttribute('aria-label',state.collapsed?'Expand sidebar':'Collapse sidebar');$('collapseSidebar').title=state.collapsed?'Expand sidebar':'Collapse sidebar';persist();
}
function search() {
 openDialog('Search conversations',(body)=>{const input=el('input',undefined,'dialog-input');input.placeholder='Search your local chats';input.setAttribute('aria-label','Search local conversations');const results=el('div',undefined,'search-results');body.append(input,results);
  const update=()=>{results.replaceChildren();const term=input.value.trim().toLowerCase();for(const c of state.chats.filter(c=>c.title.toLowerCase().includes(term)||c.messages.some(m=>m.content.toLowerCase().includes(term))))results.append(action(c.title,null,()=>{closeDialog();selectChat(c.id);}));if(!results.childElementCount)results.append(el('p','No conversations found.','dialog-copy'));};input.oninput=update;update();});
}
function createProject() {
 openDialog('Create a local project',(body,actions)=>{body.append(el('p','Projects and chats are stored only in this browser. No cloud sync or account is connected.','dialog-copy'));const input=el('input',undefined,'dialog-input');input.maxLength=40;input.required=true;input.placeholder='Project name';input.setAttribute('aria-label','Project name');body.append(input);const create=()=>{const name=input.value.trim();if(!name){input.setCustomValidity('Enter a project name.');return input.reportValidity();}if(state.projects.some(p=>p.name.toLowerCase()===name.toLowerCase())){input.setCustomValidity('A project with this name already exists.');return input.reportValidity();}const p={id:crypto.randomUUID(),name};state.projects.push(p);project=p.id;persist();closeDialog();current=null;setPage('chat');render();};input.oninput=()=>input.setCustomValidity('');actions.append(action('Cancel',null,closeDialog),action('Create project',null,create,'primary'));$('modalForm').onsubmit=e=>{e.preventDefault();create();};});
}
function exportChat() {
 const c=currentChat();if(!c)return toast('Start a conversation before exporting.');const a=el('a');const url=URL.createObjectURL(new Blob([JSON.stringify({version:2,conversation:c},null,2)],{type:'application/json'}));a.href=url;a.download='mindscape-conversation.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function renderCollection(kind) {
 const root=$('collectionContent');root.replaceChildren();
 if(kind==='library'){
  $('collectionTitle').textContent='Your library';$('collectionDescription').textContent='Answers you saved, stored in this browser. Nothing is uploaded.';
  for(const c of state.chats)for(const m of c.messages)if(m.role==='assistant'&&state.saved.includes(m.id)){const card=el('section',undefined,'collection-card');card.append(el('h2',c.title),el('p',provenance(m.result)),action('Open conversation',null,()=>selectChat(c.id),'secondary-button'));root.append(card);}
  if(!root.childElementCount)root.append(el('p','Save an answer in a conversation to find it here.','dialog-copy'));
  if(state.deleted.length){root.append(el('h2','Recently deleted'));for(const item of state.deleted){const card=el('section',undefined,'collection-card');card.append(el('h2',item.chat.title),el('p','Still stored in this browser. Restore the conversation to Recents.'),action('Restore conversation',null,()=>{state.chats.unshift(item.chat);state.saved=[...new Set([...state.saved,...item.saved])];state.deleted=state.deleted.filter(x=>x!==item);persist();renderSidebar();renderCollection('library');toast('Conversation restored.');},'secondary-button'));root.append(card);}}

 }else if(kind==='assistants'){
  $('collectionTitle').textContent='One workspace. Clear methods.';$('collectionDescription').textContent='These are real inference modes, not autonomous agents. Each question is evaluated independently.';
  for(const [mode,title,text] of [['hybrid','Hybrid','Local Qwen advisory interpretation with bounded, verified tools.'],['pretrained','Pretrained','Local Qwen only. Responses are explicitly unverified.'],['tools','Tools only','Deterministic tools without neural inference.'],['native','Native research limits','Reports the separate R6 research interface limitations.']]){const card=el('section',undefined,'collection-card');card.append(el('h2',title),el('p',text),action('Use '+title,null,()=>{$('mode').value=mode;setPage('chat');render();$('prompt').focus();},'secondary-button'));root.append(card);}
 }else{
  $('collectionTitle').textContent='A place to start';$('collectionDescription').textContent='Real examples for the supported tools. Choose one to edit it before sending.';
  for(const [key,title] of [['math','Mathematics'],['physics','Physics'],['coding','Computer science']]){const card=el('section',undefined,'collection-card');card.append(el('h2',title),el('p',prompts[key]),action('Try this question',null,()=>{setPage('chat');$('prompt').value=prompts[key];render();$('prompt').focus();},'secondary-button'));root.append(card);}
 }
}
async function research() {
 setPage('research');try{const r=await fetch('/api/capabilities');if(!r.ok)throw new Error('Capabilities unavailable.');$('capabilities').textContent=JSON.stringify(await r.json(),null,2);}catch(e){$('capabilities').textContent=e.message;}
}
// DOM bindings are installed once; rerenders replace only their own message/sidebar nodes.
hydrateIcons();document.body.classList.toggle('sidebar-collapsed',!!state.collapsed);
$('composer').onsubmit=e=>{e.preventDefault();send($('prompt').value);};$('prompt').oninput=()=>{resizePrompt();syncComposer();};$('prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();send($('prompt').value);}};
$('cancel').onclick=stop;$('newChat').onclick=newChat;$('chatTab').onclick=()=>{setPage('chat');render();};$('collapseSidebar').onclick=collapseSidebar;$('menu').onclick=openDrawer;$('drawerBackdrop').onclick=()=>{closeDrawer();$('menu').focus();};$('addProject').onclick=createProject;$('searchButton').onclick=search;
$('projectContext').onclick=()=>{project=null;render();};$('libraryPage').onclick=()=>setPage('library');$('assistantsPage').onclick=()=>setPage('assistants');$('explorePage').onclick=()=>setPage('explore');$('researchPage').onclick=research;
const workspaceInfo=()=>info('A local Mindscape workspace','Chats, projects and saved answers stay in this browser, without cloud sync. Each backend request is evaluated independently; a chat thread does not add model memory. No account, subscription, voice, attachments, scheduled tasks or Work agent is connected. Pretrained/tool assistance is separate from native Mindscape research. Do not enter credentials or private research secrets. Export important chats before clearing browser data.');
$('workspaceMenu').onclick=workspaceInfo;$('workspaceInfo').onclick=workspaceInfo;$('closeTrace').onclick=closeTrace;$('closeModal').onclick=closeDialog;
$('modal').addEventListener('close',()=>modalReturn?.focus());
$('addAction').onclick=()=>{const open=$('actionMenu').hidden;$('actionMenu').hidden=!open;$('addAction').setAttribute('aria-expanded',String(open));if(open)$('insertPython').focus();};
function insert(text){$('actionMenu').hidden=true;$('addAction').setAttribute('aria-expanded','false');$('prompt').value=text;resizePrompt();syncComposer();$('prompt').focus();}
$('insertPython').onclick=()=>insert('Predict the output:\n```python\nprint(1 + 2)\n```');$('insertJSON').onclick=()=>insert('{"subject":"math","task":"differentiate","expression":"x**3","variable":"x"}');$('export').onclick=()=>{$('actionMenu').hidden=true;$('addAction').setAttribute('aria-expanded','false');exportChat();};
document.querySelectorAll('[data-suggestion]').forEach(b=>b.onclick=()=>insert(prompts[b.dataset.suggestion]));
document.addEventListener('click',e=>{if(!$('addAction').contains(e.target)&&!$('actionMenu').contains(e.target)){$('actionMenu').hidden=true;$('addAction').setAttribute('aria-expanded','false');}});
document.addEventListener('keydown',e=>{if(e.key==='Tab'&&document.body.classList.contains('drawer-open')){const items=[...$('sidebar').querySelectorAll('button:not(:disabled),a,input')].filter(n=>n.offsetParent!==null);const first=items[0],last=items.at(-1);if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus();}}if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();newChat();}if(e.key==='Escape'){if(!$('inspector').hidden)closeTrace();if(document.body.classList.contains('drawer-open')){closeDrawer();$('menu').focus();}if(!$('actionMenu').hidden){$('actionMenu').hidden=true;$('addAction').setAttribute('aria-expanded','false');$('addAction').focus();}}});
$('history').addEventListener('keydown',e=>{if(!['ArrowDown','ArrowUp'].includes(e.key))return;const items=[...$('history').querySelectorAll('.history-item')];const i=items.indexOf(document.activeElement);if(i<0)return;e.preventDefault();items[(i+(e.key==='ArrowDown'?1:items.length-1))%items.length]?.focus();});
$('runNative').onclick=async()=>{const b=$('runNative');b.disabled=true;$('nativeResult').textContent='Running the mounted research demonstration…';try{const r=await fetch('/api/native-demo');if(!r.ok)throw new Error('Native demonstration unavailable.');$('nativeResult').textContent=JSON.stringify(await r.json(),null,2);}catch(e){$('nativeResult').textContent=e.message;}finally{b.disabled=false;}};
matchMedia('(max-width: 800px)').addEventListener('change',closeDrawer);
if(window.visualViewport){let frame;const fit=()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(()=>document.documentElement.style.setProperty('--visual-height',window.visualViewport.height+'px'));};window.visualViewport.addEventListener('resize',fit,{passive:true});fit();}
closeDrawer();render();persist();
const interrupted=currentChat()?.messages.at(-1);
if(interrupted?.state==='interrupted'){$('prompt').value=interrupted.content;resizePrompt();syncComposer();showError('The previous page lost its request connection. The server may finish that call; your message is restored.',()=>send(interrupted.content,interrupted.id));}
fetch('/api/health').then(r=>{if(!r.ok)throw new Error();return r.json();}).then(h=>{$('connection').lastChild.textContent=h.neural?.ready?'Local':'Tools available';$('connection').title=h.neural?.ready?'Local pretrained model ready':'No pretrained model loaded; tools-only mode is available';}).catch(()=>{$('connection').lastChild.textContent='Disconnected';$('connection').title='Local backend unavailable';});
