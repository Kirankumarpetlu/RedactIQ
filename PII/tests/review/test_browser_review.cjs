// Run with: node tests/review/test_browser_review.cjs
// Synthetic fixtures only; this never signs off the real 500-row dataset.
const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const template=fs.readFileSync('scripts/ground_truth_review_template.html','utf8');
const code=template.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/\nrender\(\);\s*$/,'');
const fixtures=[
  {row_index:'test-1',context:'Hi 😀 Ram',source_fingerprint:'fixture-1',language:'English',script:'Latin',source:'synthetic_fixture',suggestions:[],uncertainties:[]},
  {row_index:'test-2',context:'No PII.\nQuoted "text".',source_fingerprint:'fixture-2',language:'English',script:'Latin',source:'synthetic_fixture',suggestions:[],uncertainties:[]}
];
const nodes=new Map();
const context=vm.createContext({document:{getElementById(id){if(!nodes.has(id))nodes.set(id,{textContent:id==='data'?JSON.stringify(fixtures):'',value:''});return nodes.get(id);}},console});
vm.runInContext(code,context);
assert.equal(vm.runInContext("validTags(rows[0],[{entity_type:'PERSON',text:'Ram',start:5,end:8}]).length",context),1,'Unicode offsets must count Python characters, including emoji as one');
assert.throws(()=>vm.runInContext("validTags(rows[0],[{entity_type:'PERSON',text:'Ram',start:6,end:9}])",context),/original text/,'Reject UTF-16 offsets');
assert.throws(()=>vm.runInContext("validTags(rows[0],[{entity_type:'PERSON',text:'Ram',start:5,end:8},{entity_type:'PERSON',text:'Ram',start:5,end:8}])",context),/Duplicate/);
assert.throws(()=>vm.runInContext('reviewedCsv()',context),/exhaustive human review/,'Never export pending rows');
vm.runInContext("Object.assign(state[0],{reviewed:true,reviewer:'TEST FIXTURE',reviewed_at:'test',read:true,tags:[{entity_type:'PERSON',text:'Ram',start:5,end:8}]})",context);
assert.throws(()=>vm.runInContext('reviewedCsv()',context),/exhaustive human review/,'One completed fixture cannot stand for every row');
vm.runInContext("Object.assign(state[1],{reviewed:true,reviewer:'TEST FIXTURE',reviewed_at:'test',read:true})",context);
const csv=vm.runInContext('reviewedCsv()',context);
assert.ok(csv.startsWith('\ufeff'),'CSV should preserve Windows UTF-8 encoding');
assert.ok(csv.includes('No PII.\nQuoted ""text"".'),'Embedded newlines and quotes must be preserved');
assert.ok(csv.includes('"[]"'),'Exhaustively checked no-PII fixture should export an explicit empty list');
assert.ok(csv.includes('"reviewed","exhaustive","TEST FIXTURE"'),'Export includes reviewer metadata');
console.log('PASS: Unicode source offsets, duplicate rejection, pending and partial export blocks, and synthetic reviewed CSV escaping.');
