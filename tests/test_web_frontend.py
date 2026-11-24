"""Frontend contract tests; independent of the scientific/backend service."""
from html.parser import HTMLParser
from pathlib import Path
import json
import shutil
import subprocess

import pytest

from flask import Flask, render_template

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'src/football_predictor/web'


class Document(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.nodes = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.nodes.append((tag, dict(attrs)))

    def by_id(self, identifier):
        return next(attrs for _, attrs in self.nodes if attrs.get('id') == identifier)


def render():
    app = Flask(__name__, template_folder=str(WEB / 'templates'), static_folder=str(WEB / 'static'))
    with app.test_request_context('/'):
        return render_template('index.html', csrf_token='test-token-<escaped>')


def test_romanian_workspace_renders_accessible_navigation_and_secure_assets():
    assert (WEB / 'templates/index.html').exists(), 'The real web workspace template is missing'
    markup = render()
    doc = Document(markup)
    assert '<html lang="ro">' in markup
    assert any(tag == 'meta' and attrs.get('name') == 'csrf-token' and attrs['content'] == 'test-token-<escaped>' for tag, attrs in doc.nodes)
    assert any(tag == 'link' and attrs.get('href', '').endswith('/static/app.css') for tag, attrs in doc.nodes)
    assert any(tag == 'script' and attrs.get('src', '').endswith('/static/app.js') and 'defer' in attrs for tag, attrs in doc.nodes)
    for page in ('predict', 'experiments', 'data', 'method'):
        assert doc.by_id(f'page-{page}')
        assert any(tag == 'a' and attrs.get('href') == f'#{page}' for tag, attrs in doc.nodes)
    assert doc.by_id('status-message')['aria-live'] == 'polite'
    assert 'disabled' in doc.by_id('train-button')
    assert not any('style' in attrs or any(key.startswith('on') for key in attrs) for _, attrs in doc.nodes)


def test_prediction_and_import_forms_have_labels_and_disabled_until_ready_controls():
    doc = Document(render())
    labels = {attrs.get('for') for tag, attrs in doc.nodes if tag == 'label'}
    for identifier in ('home-team', 'away-team', 'fixture-date', 'upload-file', 'season-select', 'history-team'):
        assert identifier in labels, f'{identifier} requires a visible label'
        assert doc.by_id(identifier)
    assert doc.by_id('fixture-date')['type'] == 'date'
    assert 'disabled' in doc.by_id('predict-button')
    assert doc.by_id('upload-file')['name'] == 'file'
    assert doc.by_id('upload-form')['enctype'] == 'multipart/form-data'
    assert doc.by_id('features-content')
    assert any(tag == 'details' for tag, _ in doc.nodes)
    assert any(tag == 'table' and attrs.get('id') == 'probability-table' for tag, attrs in doc.nodes)


NODE_SETUP = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM} = require('jsdom');
const dom = new JSDOM(MARKUP, {url: 'http://localhost/', runScripts: 'outside-only'});
const w = dom.window;
const doc = w.document;
const $ = id => doc.getElementById(id);
const calls = [];
const state = {ready:true, trained:false, busy:false, active_job:null, error:null,
 data:{name:'istoric.csv', n_matches:120, n_teams:2, start_date:'2020-01-01',
 end_date:'2025-05-01', teams:['Alpha','Beta'], seasons:[2025,2024],default_prediction_date:'2025-05-02'}};
const reply = (payload, status=200) => ({ok:status<400,status,json:async()=>payload});
let route = async (url, options) => {
 if(url === '/api/state') return reply(state);
 if(url === '/api/report') return reply({metrics:[],confusion:{},tuning:[]});
 if(url.startsWith('/api/standings')) return reply({rows:[]});
 if(url.startsWith('/api/history')) return reply({rows:[],total:0});
 throw Error('Unexpected request: '+url);
};
w.fetch = async (url, options={}) => {calls.push([url,options]); return route(url,options);};
const tick = () => new Promise(resolve => setTimeout(resolve, 15));
const allText = id => $(id).textContent;
"""


def run_frontend(script, before=''):
    assert (WEB / 'static/app.js').exists(), 'The application controller is missing'
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for DOM interaction tests')
    probe = subprocess.run([node, '-e', "require('jsdom')"], capture_output=True, text=True)
    if probe.returncode:
        pytest.skip('jsdom is needed for DOM interaction tests')
    code = 'const MARKUP = ' + json.dumps(render()) + ';\n' + NODE_SETUP
    code += before
    code += '\nconst SOURCE = ' + json.dumps((WEB / 'static/app.js').read_text()) + '; w.eval(SOURCE);\n'
    code += '(async()=>{await tick();' + script + '\n w.close();})().catch(e=>{console.error(e);w.close();process.exitCode=1;});'
    result = subprocess.run([node, '-e', code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_state_populates_real_dataset_and_hash_navigation():
    run_frontend(r"""
 assert.equal($('train-button').disabled, false);
 assert.equal($('predict-button').disabled, true);
 assert.equal($('home-team').options[0].textContent, 'Alpha');
 assert.equal($('away-team').value, 'Beta');
 assert.equal($('fixture-date').value, '2025-05-02');
 assert.match(allText('dataset-summary'), /120/);
 assert.match(allText('dataset-warning'), /2025-05-01/);
 w.location.hash = '#experiments'; await tick();
 assert.equal($('page-experiments').hidden,false);
 assert.equal($('page-predict').hidden,true);
 assert.equal(doc.querySelector('nav a[href="#experiments"]').getAttribute('aria-current'),'page');
 """)


def test_styles_support_responsive_focus_reduced_motion_and_csp_safe_heatmap():
    path = WEB / 'static/app.css'
    assert path.exists(), 'Responsive workspace styles are missing'
    css = path.read_text()
    for rule in ('@media', 'prefers-reduced-motion', ':focus-visible', '[hidden]', 'overflow-x: auto', 'min-height: 44px', '[aria-current="page"]'):
        assert rule in css
    for bucket in range(6):
        assert f'.heat-{bucket}' in css
    assert '@import' not in css and 'url(http' not in css


def test_prediction_posts_fixture_and_renders_adaptive_matrix_safely():
    run_frontend(r"""
 const previous = route;
 const matrix = Array.from({length:12}, (_,i) => Array.from({length:11}, (_,j) => i===0 && j===0 ? 0.25 : 0.001));
 route = async (url,opt) => {
  if(url === '/api/predict') return reply({job_id:'predict-1'},202);
  if(url === '/api/jobs/predict-1') return reply({kind:'predict',status:'complete',progress:1,
   result:{probabilities:{KNN:[0.5,0.3,0.2],RF:[0.4,0.3,0.3],Poisson:[0.45,0.3,0.25]},
    scores:[{score:'1-0',probability:0.2}],score_matrix:matrix,expected_goals:[1.6,1.1],
    features:{'<img src=x onerror=alert(1)>':2.5},score_tail:0.003}});
  return previous(url,opt);
 };
 $('fixture-date').value='2025-06-10';
 $('prediction-form').dispatchEvent(new w.Event('submit',{cancelable:true})); await tick(); await tick();
 const post=calls.find(([url])=>url==='/api/predict'); assert.ok(post);
 assert.deepEqual(JSON.parse(post[1].body),{home:'Alpha',away:'Beta',date:'2025-06-10'});
 assert.equal(post[1].credentials,'same-origin');
 assert.equal(post[1].headers['X-CSRF-Token'],'test-token-<escaped>');
 assert.equal($('prediction-output').hidden,false); assert.equal($('prediction-empty').hidden,true);
 assert.equal($('probability-body').children.length,3);
 assert.match(allText('probability-body'),/50,0 %/);
 assert.match(allText('fixture-label'),/Alpha.*Beta/);
 assert.match(allText('expected-goals'),/1,60/);
 assert.match(allText('score-list'),/1-0.*20,0 %/);
 assert.equal($('score-matrix').querySelectorAll('tbody tr').length,12);
 assert.equal($('score-matrix').querySelectorAll('tbody td').length,132);
 assert.ok($('score-matrix').querySelector('.heat-5')); assert.ok($('score-matrix').querySelector('.heat-0'));
 assert.match(allText('features-content'),/<img/); assert.equal($('features-content').querySelector('img'),null);
 assert.equal(doc.querySelector('[style]'),null);
 """, before="state.trained=true;")


@pytest.mark.parametrize('home,away,date', [
    ('Alpha', 'Alpha', '2025-05-02'), ('', 'Beta', '2025-05-02'),
    ('Alpha', 'Beta', '2025-05-01'), ('Alpha', 'Beta', ''),
])
def test_invalid_fixture_never_posts_even_on_programmatic_submit(home, away, date):
    run_frontend('''
 $('home-team').value=HOME; $('away-team').value=AWAY; $('fixture-date').value=DATE;
 $('fixture-date').dispatchEvent(new w.Event('change'));
 assert.equal($('predict-button').disabled,true);
 $('prediction-form').dispatchEvent(new w.Event('submit',{cancelable:true})); await tick();
 assert.ok(!calls.some(([url])=>url==='/api/predict'));
 assert.match(allText('status-message'),/echip|dat/i);
 assert.equal($('fixture-date').min,'2025-05-02');
 '''.replace('HOME', json.dumps(home)).replace('AWAY', json.dumps(away)).replace('DATE', json.dumps(date)), before='state.trained=true;')


def test_upload_is_multipart_and_failure_preserves_source_report_and_prediction():
    run_frontend(r"""
 $('report-content').textContent='Evaluare existentă'; $('prediction-output').hidden=false;
 $('prediction-empty').hidden=true; $('fixture-label').textContent='Alpha — Beta';
 const previous=route;
 route=async(url,opt)=> {
  if(url==='/api/upload') return reply({job_id:'upload-bad'},202);
  if(url==='/api/jobs/upload-bad') return reply({kind:'upload',status:'error',progress:1,error:'CSV invalid <script>x</script>'});
  return previous(url,opt);
 };
 const file=new w.File(['bad'], 'istoric.csv',{type:'text/csv'});
 Object.defineProperty($('upload-file'),'files',{value:[file],configurable:true});
 $('upload-form').dispatchEvent(new w.Event('submit',{cancelable:true})); await tick(); await tick();
 const post=calls.find(([url])=>url==='/api/upload'); assert.ok(post);
 assert.equal(post[1].body.get('file').name,'istoric.csv');
 assert.equal(post[1].headers['Content-Type'],undefined);
 assert.equal(post[1].headers['X-CSRF-Token'],'test-token-<escaped>');
 assert.match(allText('status-message'),/CSV invalid/); assert.equal($('status-message').querySelector('script'),null);
 assert.equal(allText('report-content'),'Evaluare existentă');
 assert.equal($('prediction-output').hidden,false); assert.match(allText('dataset-summary'),/istoric.csv/);
 assert.equal($('predict-button').disabled,false); assert.equal($('job-progress').hidden,true);
 """, before='state.trained=true;')


@pytest.mark.parametrize('operation', ['upload', 'reset'])
def test_successful_source_operation_clears_old_models_report_results_and_date(operation):
    run_frontend(r"""
 $('report-content').textContent='RAPORT VECHI'; $('export-links').textContent='EXPORT VECHI';
 $('prediction-output').hidden=false; $('prediction-empty').hidden=true;
 $('fixture-date').value='2028-01-01';
 const previous=route;
 route=async(url,opt)=> {
  if(url==='/api/OPERATION') return reply({job_id:'source-1'},202);
  if(url==='/api/jobs/source-1') {
   state.trained=false; state.data={...state.data,name:'nou.csv',end_date:'2026-06-01',default_prediction_date:'2026-06-02'};
   return reply({kind:'OPERATION',status:'complete',progress:1,result:state});
  }
  return previous(url,opt);
 };
 if('OPERATION'==='upload') {
  Object.defineProperty($('upload-file'),'files',{value:[new w.File(['csv'],'nou.csv')],configurable:true});
  $('upload-form').dispatchEvent(new w.Event('submit',{cancelable:true}));
 } else { assert.ok($('reset-button')); $('reset-button').click(); }
 await tick(); await tick();
 assert.ok(calls.some(([url,opt])=>url==='/api/OPERATION' && opt.headers['X-CSRF-Token']==='test-token-<escaped>'));
 assert.equal($('predict-button').disabled,true); assert.equal($('train-button').disabled,false);
 assert.equal($('prediction-output').hidden,true); assert.equal($('prediction-empty').hidden,false);
 assert.doesNotMatch(allText('report-content'),/RAPORT VECHI/); assert.equal(allText('export-links'),'');
 assert.equal($('fixture-date').value,'2026-06-02'); assert.equal($('fixture-date').min,'2026-06-02');
 assert.match(allText('dataset-summary'),/nou.csv/);
 """.replace('OPERATION', operation), before='state.trained=true;')


def test_export_only_exposes_whitelisted_backend_downloads():
    run_frontend(r"""
 const previous=route, id='0123456789abcdef0123456789abcdef';
 route=async(url,opt)=> {
  if(url==='/api/export') return reply({job_id:'export-1'},202);
  if(url==='/api/jobs/export-1') return reply({kind:'export',status:'complete',progress:1,result:{export_id:id,
   files:[{name:'metrics.csv',url:`/api/exports/${id}/metrics.csv`},
    {name:'report.zip',url:`/api/exports/${id}/report.zip`},
    {name:'report.json',url:'https://evil.invalid/report.json'},
    {name:'tuning.csv',url:`/api/exports/${id}/../tuning.csv`},
    {name:'bad.html',url:`/api/exports/${id}/bad.html`}]}});
  return previous(url,opt);
 };
 $('export-button').click(); await tick(); await tick();
 const post=calls.find(([url])=>url==='/api/export'); assert.ok(post); assert.equal(post[1].body,'{}');
 assert.equal(post[1].headers['X-CSRF-Token'],'test-token-<escaped>');
 const links=$('export-links').querySelectorAll('a'); assert.equal(links.length,2);
 assert.equal(links[0].getAttribute('href'),`/api/exports/${id}/metrics.csv`);
 assert.equal(links[1].textContent,'report.zip'); assert.ok(links[0].hasAttribute('download'));
 """,before='state.trained=true;')


def test_data_tables_render_actual_rows_and_ignore_out_of_order_filters():
    run_frontend(r"""
 assert.match(allText('standings-content'),/Alpha/); assert.match(allText('standings-content'),/17/);
 assert.match(allText('history-content'),/2025-04-01/); assert.match(allText('history-total'),/1 din 39/);
 let oldResolve; const previous=route;
 route=async(url,opt)=>{
  if(url==='/api/history?team=Beta&limit=20') return new Promise(resolve=>oldResolve=resolve);
  if(url==='/api/history?team=Alpha&limit=20') return reply({rows:[{date:'2025-04-02',home_team:'ACTUAL',away_team:'Beta',home_goals:2,away_goals:0}],total:1});
  return previous(url,opt);
 };
 $('history-team').value='Beta'; $('history-team').dispatchEvent(new w.Event('change')); await tick();
 assert.match(allText('history-content'),/încarcă/);
 $('history-team').value='Alpha'; $('history-team').dispatchEvent(new w.Event('change')); await tick();
 oldResolve(reply({rows:[{date:'2025-04-01',home_team:'ÎNTÂRZIAT',away_team:'Alpha',home_goals:0,away_goals:1}],total:1})); await tick();
 assert.match(allText('history-content'),/ACTUAL/); assert.doesNotMatch(allText('history-content'),/ÎNTÂRZIAT/);
 $('season-select').value='2024'; $('season-select').dispatchEvent(new w.Event('change')); await tick();
 assert.ok(calls.some(([url])=>url==='/api/standings?season=2024'));
 """, before=r"""
 const originalRoute=route;
 route=async(url,opt)=> {
  if(url.startsWith('/api/standings')) return reply({rows:[{position:1,team:'Alpha',played:8,points:17,gf:16,ga:5,gd:11}]});
  if(url.startsWith('/api/history')) return reply({rows:[{date:'2025-04-01',home_team:'Alpha',away_team:'Beta',home_goals:2,away_goals:1}],total:39});
  return originalRoute(url,opt);
 };
 """)


def test_reload_resumes_active_job_with_live_progress_and_reasonable_polling():
    run_frontend(r"""
 for(let i=0;i<20 && polls<3;i++) await tick();
 assert.ok(calls.some(([url])=>url==='/api/jobs/resume-1'));
 assert.deepEqual(delays,[700,700]);
 assert.match(allText('report-content'),/RELUAT/); assert.equal($('job-progress').hidden,true);
 assert.equal($('train-button').disabled,false); assert.equal($('predict-button').disabled,false);
 """,before=r"""
 state.busy=true; state.active_job='resume-1'; let polls=0; const delays=[];
 const originalTimeout=w.setTimeout.bind(w); w.setTimeout=(fn,ms)=>{delays.push(ms); return originalTimeout(fn,0);};
 const originalRoute=route;
 route=async(url,opt)=> {
  if(url==='/api/jobs/resume-1') {
   polls++; if(polls<3) return reply({id:'resume-1',kind:'train',status:polls===1?'queued':'running',progress:0.4,message:'Se antrenează'});
   state.busy=false; state.active_job=null; state.trained=true;
   return reply({id:'resume-1',kind:'train',status:'complete',progress:1,result:{metrics:[{model:'RELUAT',accuracy:.5}],confusion:{},tuning:[]}});
  }
  return originalRoute(url,opt);
 };
 """)


def test_refresh_fetches_existing_report_retains_selection_until_source_changes():
    run_frontend(r"""
 assert.match(allText('report-content'),/RAPORT SERVER/);
 $('home-team').value='Beta'; $('away-team').value='Alpha'; $('fixture-date').value='2027-01-03';
 $('refresh-button').click(); await tick(); await tick();
 assert.equal($('home-team').value,'Beta'); assert.equal($('away-team').value,'Alpha');
 assert.equal($('fixture-date').value,'2027-01-03');
 state.trained=false; state.data={...state.data,name:'schimbat.csv',end_date:'2026-02-01',default_prediction_date:'2026-02-02'};
 $('refresh-button').click(); await tick(); await tick();
 assert.doesNotMatch(allText('report-content'),/RAPORT SERVER/); assert.equal($('fixture-date').value,'2026-02-02');
 assert.match(allText('dataset-warning'),/învechit|vechi/i);
 """,before=r"""
 state.trained=true; const originalRoute=route;
 route=async(url,opt)=>url==='/api/report'?reply({metrics:[{model:'RAPORT SERVER',accuracy:.6}],confusion:{},tuning:[]}):originalRoute(url,opt);
 """)


@pytest.mark.parametrize('failure', ['csrf', 'http', 'network', 'nonjson', 'job'])
def test_async_failures_are_visible_recoverable_and_never_fabricate_results(failure):
    run_frontend(r"""
 const previous=route;
 route=async(url,opt)=>{
  if(url==='/api/train') {
   if('FAILURE'==='csrf') return reply({error:'Forbidden'},403);
   if('FAILURE'==='http') return reply({error:'Date insuficiente'},422);
   if('FAILURE'==='network') throw new Error('offline');
   if('FAILURE'==='nonjson') return {ok:false,status:502,json:async()=>{throw new Error('Unexpected token <');}};
   return reply({job_id:'error-1'},202);
  }
  if(url==='/api/jobs/error-1') return reply({kind:'train',status:'error',error:'Antrenarea a eșuat',progress:0.4});
  return previous(url,opt);
 };
 $('train-button').click(); await tick(); await tick();
 assert.ok($('job-status').classList.contains('error'));
 assert.match(allText('status-message'), /SECURITY|Date insuficiente|offline|răspuns invalid|Antrenarea a eșuat/i);
 assert.equal($('train-button').disabled,false); assert.equal($('job-progress').hidden,true);
 assert.equal($('predict-button').disabled,true); assert.equal($('prediction-output').hidden,true);
 assert.equal($('report-content').querySelector('table'),null);
 """.replace('FAILURE', failure).replace('SECURITY','securitate'))


def test_new_job_cannot_be_overwritten_by_stale_refresh_report():
    run_frontend(r"""
 let resolveOld; const previous=route;
 route=async(url,opt)=>{
  if(url==='/api/report') return new Promise(resolve=>resolveOld=resolve);
  if(url==='/api/train') return reply({job_id:'train-new'},202);
  if(url==='/api/jobs/train-new') return reply({kind:'train',status:'complete',progress:1,result:{metrics:[{model:'NOU',accuracy:.5}],tuning:[],confusion:{}}});
  return previous(url,opt);
 };
 $('refresh-button').click(); await tick();
 $('train-button').click(); await tick(); await tick();
 resolveOld(reply({metrics:[{model:'ÎNVECHIT',accuracy:.2}],confusion:{},tuning:[]})); await tick();
 assert.match(allText('report-content'),/NOU/); assert.doesNotMatch(allText('report-content'),/ÎNVECHIT/);
 """,before='state.trained=true;')


def test_reload_keeps_fixture_preferences_only_for_the_same_dataset():
    run_frontend(r"""
 $('home-team').value='Beta'; $('away-team').value='Alpha'; $('fixture-date').value='2027-02-02';
 $('fixture-date').dispatchEvent(new w.Event('change'));
 async function reload() {
  const next=new JSDOM(MARKUP,{url:'http://localhost/',runScripts:'outside-only'});
  for(let i=0;i<w.localStorage.length;i++) {const k=w.localStorage.key(i);next.window.localStorage.setItem(k,w.localStorage.getItem(k));}
  next.window.fetch=w.fetch; next.window.eval(SOURCE); await tick(); await tick(); return next;
 }
 const same=await reload();
 try {
  assert.equal(same.window.document.getElementById('home-team').value,'Beta');
  assert.equal(same.window.document.getElementById('away-team').value,'Alpha');
  assert.equal(same.window.document.getElementById('fixture-date').value,'2027-02-02');
 } finally {same.window.close();}
 state.data={...state.data,name:'diferit.csv'};
 const changed=await reload();
 try {assert.equal(changed.window.document.getElementById('fixture-date').value,'2025-05-02');}
 finally {changed.window.close();}
 """,before='state.trained=true;')


def test_scroll_regions_and_prediction_validation_are_keyboard_accessible():
    doc = Document(render())
    for _, attrs in doc.nodes:
        if 'table-scroll' in attrs.get('class', '').split():
            assert attrs.get('tabindex') == '0', 'Horizontal data regions need keyboard scrolling'
    for identifier in ('home-team', 'away-team', 'fixture-date'):
        assert doc.by_id(identifier).get('aria-describedby') == 'prediction-hint'
    css = (WEB / 'static/app.css').read_text()
    assert 'overflow-wrap: anywhere' in css
    assert '#reset-button' in css


def test_train_posts_csrf_polls_job_and_renders_actual_report():
    run_frontend(r"""
 const report = {metrics:[{model:'KNN',accuracy:0.5,macro_f1:0.4,log_loss:1.1,brier:0.7,n_test:24}],
 tuning:[{model:'KNN',params:{n_neighbors:7},validation_log_loss:1.2}],
 split:{train:{n:72,start:'2020-01-01',end:'2023-01-01'},validation:{n:24,start:'2023-01-02',end:'2024-01-01'},test:{n:24,start:'2024-01-02',end:'2025-05-01'},protocol:'walk-forward test'},
 confusion:{KNN:[[2,1,0],[1,3,1],[0,2,4]]}};
 const previous = route;
 route = async (url,opt) => {
  if(url === '/api/train') return reply({job_id:'train-1'},202);
  if(url === '/api/jobs/train-1') {state.trained=true; return reply({id:'train-1',kind:'train',status:'complete',progress:1,message:'Antrenare completă',result:report});}
  if(url === '/api/report') return reply(report);
  return previous(url,opt);
 };
 $('train-button').click(); await tick(); await tick();
 const post = calls.find(([url]) => url === '/api/train');
 assert.ok(post, 'Train must call the real API');
 assert.equal(post[1].headers['X-CSRF-Token'],'test-token-<escaped>');
 assert.equal(post[1].method,'POST'); assert.equal(post[1].body,'{}');
 assert.ok(calls.some(([url]) => url === '/api/jobs/train-1'));
 assert.match(allText('report-content'),/KNN/);
 assert.match(allText('report-content'),/50[,.]0/);
 assert.match(allText('report-content'),/walk-forward test/);
 assert.match(allText('report-content'),/n_neighbors/);
 assert.ok($('report-content').querySelector('table[aria-label="KNN · real pe rânduri, prezis pe coloane"]'));
 assert.equal($('predict-button').disabled,false);
 assert.equal($('job-progress').hidden,true);
 """)


