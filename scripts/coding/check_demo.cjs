/* Execute the exported playback JavaScript against a minimal test DOM. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('demo/coding/index.html', 'utf8');
const evidence = JSON.parse(fs.readFileSync('demo/coding/evidence.json','utf8'));
assert.equal(evidence.length, 200);
const elements = new Map();
function element() {
  return {value: '0', max: 0, textContent: '', children: [],
    classList: {toggle() {}},
    append(...nodes) { this.children.push(...nodes); if (this.children.length === nodes.length && nodes[0]?.value) this.value = nodes[0].value; },
    replaceChildren(...nodes) { this.children = nodes; }};
}
const document = {getElementById(id) { if (!elements.has(id)) elements.set(id,element()); return elements.get(id); }, createElement: element};
const context = vm.createContext({document});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
const node = id => document.getElementById(id);
assert.equal(node('task').children.length,100);
for (const r of evidence) {
  node('task').value=r.task_id;
  node(r.condition==='model_only'?'plain':'mind').onclick();
  assert.equal(node('counter').textContent,`0 / ${r.steps.length}`);
  for (let i=0;i<=r.steps.length;i++) {
    node('step').value=String(i);node('step').oninput();
    const files=i ? r.steps[i-1].files : r.before;
    assert.equal(node('after').children.length,Object.keys(files).length*2);
    const rendered=node('after').children.filter((_,index)=>index%2===1).map(n=>n.textContent);
    assert.deepEqual(rendered,Object.values(files));
    assert.equal(node('events').children.length,i);
    const state=JSON.parse(node('state').textContent);
    if (i===r.steps.length) {
      assert.equal(state.verification.terminal_goal,r.success);
      assert.equal(node('goal').textContent,`${r.success?'Passed':'Failed'} ${r.terminal_passed}/${r.terminal_total}`);
    } else {
      assert.equal(node('goal').textContent,'Pending');
      assert.equal(state.verification,'Private terminal assessment pending');
    }
  }
  node('forward').onclick();assert.equal(+node('step').value,r.steps.length);
  node('back').onclick();assert.equal(+node('step').value,Math.max(0,r.steps.length-1));
}
assert.ok(!html.includes('ground_truth_patch'));
assert.ok(!html.includes('hidden_tests'));
assert.ok(!html.includes('correct_repository'));
console.log('PASS: all 200 playback records, every step, controls and private-field exclusion');
