/* Native MathML rendering for a conservative LaTeX subset. Never interprets HTML. */
'use strict';
window.renderMind1Math=function(root){
 const ns='http://www.w3.org/1998/Math/MathML';
 const node=(name,text)=>{const n=document.createElementNS(ns,name);if(text!==undefined)n.textContent=text;return n};
 const symbols={alpha:'α',beta:'β',gamma:'γ',delta:'δ',theta:'θ',lambda:'λ',mu:'μ',pi:'π',sigma:'σ',phi:'φ',omega:'ω',Delta:'Δ',Sigma:'Σ',Omega:'Ω',infty:'∞',times:'×',cdot:'·',pm:'±',leq:'≤',geq:'≥',neq:'≠',approx:'≈',sum:'∑',prod:'∏',int:'∫',partial:'∂',nabla:'∇',to:'→',in:'∈'};
 function parse(src){if(src.length>4000)throw Error('Expression too long');let i=0,count=0;
  function row(end,depth){if(depth>40)throw Error('Nesting limit');const r=node('mrow');while(i<src.length&&src[i]!==end){if(/\s/.test(src[i])){i++;continue}let a=atom(depth+1);while(src[i]==='^'||src[i]==='_'){const type=src[i++]==='^'?'msup':'msub',script=node(type);script.append(a,atom(depth+1));a=script}r.append(a)}if(end){if(src[i]!==end)throw Error('Unclosed group');i++}return r}
  function atom(depth){if(++count>2000||depth>40)throw Error('Complexity limit');while(/\s/.test(src[i]||'')&&i<src.length)i++;const c=src[i++];if(c===undefined)throw Error('Missing operand');if(c==='{')return row('}',depth);if(c==='\\'){const match=src.slice(i).match(/^[A-Za-z]+/);if(!match){return node('mo',src[i++]||' ')}const cmd=match[0];i+=cmd.length;if(cmd==='frac'){const n=node('mfrac');n.append(atom(depth+1),atom(depth+1));return n}if(cmd==='sqrt'){const n=node('msqrt');n.append(atom(depth+1));return n}if(cmd==='left'||cmd==='right')return atom(depth+1);if(cmd==='text'||cmd==='mathrm'){if(src[i++]!=='{')throw Error('Expected text group');const end=src.indexOf('}',i);if(end<0)throw Error('Unclosed text');const n=node('mtext',src.slice(i,end));i=end+1;return n}if(['sin','cos','tan','log','ln','exp','lim'].includes(cmd)){const n=node('mi',cmd);n.setAttribute('mathvariant','normal');return n}if(cmd==='boxed')return atom(depth+1);if(!symbols[cmd])throw Error('Unsupported command');return node('mo',symbols[cmd])}if(/[0-9]/.test(c)){const rest=src.slice(i).match(/^[0-9.]*/)[0];i+=rest.length;return node('mn',c+rest)}return node(/[A-Za-z]/.test(c)?'mi':'mo',c)}
  return row(null,0);
 }
 const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);const texts=[];while(walker.nextNode()){const t=walker.currentNode;if(!t.parentElement.closest('code,pre,a,math'))texts.push(t)}
 for(const t of texts){const pattern=/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^$\n]+?)\$/g;let last=0,match;const f=document.createDocumentFragment();while((match=pattern.exec(t.data))){f.append(document.createTextNode(t.data.slice(last,match.index)));try{const math=node('math');math.setAttribute('display',match[1]||match[2]?'block':'inline');math.setAttribute('aria-label',match[1]||match[2]||match[3]||match[4]);math.append(parse(match[1]||match[2]||match[3]||match[4]));f.append(math)}catch{f.append(document.createTextNode(match[0]))}last=pattern.lastIndex}if(last){f.append(document.createTextNode(t.data.slice(last)));t.replaceWith(f)}}
};

/* Optional view of an exact solver expression. Parsing only; never eval or arithmetic. */
window.renderMindscapeExpression=function(source){
 const ns='http://www.w3.org/1998/Math/MathML';const make=(tag,text)=>{const n=document.createElementNS(ns,tag);if(text!==undefined)n.textContent=text;return n};
 if(typeof source!=='string'||source.length>1500)return null;
 const tokens=source.match(/\*\*|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|[+\-*/^(),]/g)||[];
 if(tokens.join('')!==source.replace(/\s/g,'')||tokens.length>300)return null;let i=0,depth=0;
 const row=(...nodes)=>{const n=make('mrow');n.append(...nodes);return n};
 function atom(){if(++depth>35)throw Error('Bounded expression');let n,t=tokens[i++];if(t==='('){const value=add();if(tokens[i++]!==')')throw Error('Unclosed group');n=row(make('mo','('),value,make('mo',')'));}else if(/^\d/.test(t||''))n=make('mn',t);else if(/^[A-Za-z_]\w*$/.test(t||'')){
  if(tokens[i]==='('){if(!['sin','cos','tan','exp','log','sqrt','Abs'].includes(t))throw Error('Unsupported function');i++;const value=add();if(tokens[i++]!==')')throw Error('Unclosed function');if(t==='sqrt'){n=make('msqrt');n.append(value);}else {const name=make('mi',t);name.setAttribute('mathvariant','normal');n=row(name,make('mo','('),value,make('mo',')'));}}
  else n=make('mi',t==='pi'?'π':t);
 }else throw Error('Unsupported token');depth--;return n;}
 function power(){let n=atom();if(['**','^'].includes(tokens[i])){i++;const p=make('msup');p.append(n,unary());n=p;}return n;}
 function unary(){if(['+','-'].includes(tokens[i])){const sign=tokens[i++];return row(make('mo',sign==='-'?'−':'+'),unary());}return power();}
 function term(){let n=unary();while(['*','/'].includes(tokens[i])){const op=tokens[i++],b=unary();if(op==='/'){const f=make('mfrac');f.append(n,b);n=f;}else n=row(n,make('mo','·'),b);}return n;}
 function add(){let n=term();while(['+','-'].includes(tokens[i])){const op=tokens[i++];n=row(n,make('mo',op==='-'?'−':'+'),term());}return n;}
 try{const value=add();if(i!==tokens.length)return null;const math=make('math');math.setAttribute('display','block');math.setAttribute('aria-label',source);math.append(value);return math;}catch{return null;}
};
