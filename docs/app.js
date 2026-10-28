'use strict';
const repo='https://github.com/wang2025-source/CohereFusion';
const texts={
en:{skip:'Skip to content',navMethod:'Approach',navExplore:'Visuals',navBench:'Evaluation',navStart:'Get started',heroTitle:'Two modalities.<br><em>One coherent vision.</em>',heroDesc:'Infrared perception. Visible detail. Language guidance. Bringing complementary information together in one coherent image.',exploreButton:'Explore the visuals',sourceButton:'View source code',heroCaption:'Text-conditioned infrared & visible image fusion',visualIntro:'Different perspectives. Shared clarity.',visualBottom:'Keep the subject. Keep the context.',scroll:'Discover the approach',factModal:'Complementary modalities',factTasks:'Degradation tasks',factFigures:'Original research figures',factRecords:"Cross-dataset evaluations",methodKicker:'From complementary inputs to coherent expression',methodTitle:'The right information.<br>In the right place.',methodDesc:'Global and semantic expert branches work together through text-conditioned routing and source-constrained detail refinement to organize structure, contrast and semantic information.',architectureCaption:'Overall network architecture',step1Title:'Complementary expert branches',step1Desc:'A global expert branch models imaging information, while a semantic expert branch introduces text and higher-level features into the fusion representation.',step1Link:'Explore the branches ↗',step2Title:'Text-conditioned frequency routing',step2Desc:'Text conditioning and low-, mid- and high-frequency decomposition organize information from both branches across space and frequency.',step2Link:'Explore the routing module ↗',step3Title:'Constrained detail restoration',step3Desc:'The SCR refinement module predicts contrast gain, detail gates and source-edge routing to refine luminance after fusion.',step3Link:'Read the implementation ↗',galleryKicker:'A closer look at what is preserved',galleryTitle:'More than a glance.<br>Worth a closer look.',galleryDesc:'Explore architecture diagrams, scene comparisons and text illustrations. Open any figure for a closer look or access its full-resolution original PNG.',all:'All figures',results:'Comparisons',structure:'Architecture',analysis:'Analysis',benchKicker:"Table III · Cross-dataset evaluation",benchTitle:"Cross-dataset<br>results.",benchDesc:"All methods and six metrics on MSRS, RoadScene and LLVIP, as reported in manuscript Table III.",tableCaption:"Cross-dataset method comparison from manuscript Table III",loading:'Loading records…',startKicker:'From understanding to experimentation',startTitle:'Keep the research<br>moving forward.',startDesc:'Understand the model through its source, then experiment from traceable configurations. Training, inference and evaluation in one repository.',downloadCode:'Download code',requirements:'Environment<br>Python 3.12 · PyTorch 2.5.1 · torchvision 0.20.1',install:'Setup',infer:'Inference',train:'Training',evaluate:'Evaluation',copy:'Copy command',originals:'Original figures ↗',footerNote:'CohereFusion · GEWB / SEWB · RSFR · SCR',lightboxHint:'Arrow keys to navigate · Esc to close · Originals open at full resolution',viewOriginal:'Open original PNG ↗'},
zh:{}};
document.querySelectorAll('[data-i18n]').forEach(el=>{texts.zh[el.dataset.i18n]=el.innerHTML;});
let lang='zh',filter='all',dataset='msrs',command='setup',currentFigure=7,comparison=null;
try{lang=localStorage.getItem('cohere-language')==='en'?'en':'zh';}catch{}
const figures=[
 {
  "id": 7,
  "paperFigure": 6,
  "category": "results",
  "zh": "跨数据集融合对比",
  "en": "Cross-dataset fusion comparisons"
 },
 {
  "id": 6,
  "paperFigure": 5,
  "category": "results",
  "zh": "四种退化场景融合对比",
  "en": "Fusion under four degradation scenarios"
 },
 {
  "id": 12,
  "paperFigure": 1,
  "category": "method",
  "zh": "退化条件下的融合范式",
  "en": "Fusion paradigms under degradation"
 },
 {
  "id": 3,
  "paperFigure": 2,
  "category": "method",
  "zh": "整体网络结构",
  "en": "Overall architecture"
 },
 {
  "id": 4,
  "paperFigure": 3,
  "category": "method",
  "zh": "全局与语义专家分支",
  "en": "Global and semantic expert branches"
 },
 {
  "id": 5,
  "paperFigure": 4,
  "category": "method",
  "zh": "可靠性路由与源约束细节恢复",
  "en": "Reliability-aware routing and source-constrained refinement"
 },
 {
  "id": 8,
  "paperFigure": 8,
  "category": "analysis",
  "zh": "文本消融 · 红外噪声",
  "en": "Text ablation · infrared noise"
 },
 {
  "id": 9,
  "paperFigure": 9,
  "category": "analysis",
  "zh": "文本消融 · 可见光过曝光",
  "en": "Text ablation · visible overexposure"
 },
 {
  "id": 10,
  "paperFigure": 10,
  "category": "analysis",
  "zh": "组件消融 · 可见光弱光",
  "en": "Component ablation · visible low light"
 },
 {
  "id": 11,
  "paperFigure": 11,
  "category": "analysis",
  "zh": "组件消融 · 红外低对比",
  "en": "Component ablation · infrared low contrast"
 },
 {
  "id": 13,
  "paperFigure": 16,
  "category": "results",
  "zh": "M3FD 目标检测对比",
  "en": "Object detection comparisons on M3FD"
 },
 {
  "id": 14,
  "paperFigure": 17,
  "category": "results",
  "zh": "MSRS 语义分割对比",
  "en": "Semantic segmentation comparisons on MSRS"
 }
];
const commands={setup:'git clone https://github.com/wang2025-source/CohereFusion.git\ncd CohereFusion\n\n# Install a compatible PyTorch / torchvision pair first\npip install -r requirements.txt',infer:'python launch.py infer \\\n  --dataset-root /path/to/EMS_lite_with_text \\\n  --weights /path/to/coherefusion.pth \\\n  --output outputs/coherefusion',train:'python launch.py train \\\n  --config configs/train.json \\\n  --dataset-root /path/to/EMS_lite_with_text \\\n  --output-dir ./new_runs \\\n  --run-name coherefusion_new --split-manifest /path/to/splits.json \\\n  --epochs 40 --batch-size 8 --resume ""',evaluate:'python launch.py evaluate \\\n  --source /path/to/EMS_lite_with_text \\\n  --fused outputs/coherefusion \\\n  --output outputs/metrics.json \\\n  --workers 4 --clean-reference'};
const pad=id=>String(id).padStart(2,'0');
const original=id=>`${repo}/raw/refs/heads/main/figures/figure-${pad(id)}.png`;
function setLanguage(){
 document.documentElement.lang=lang==='zh'?'zh-CN':'en';
 document.querySelectorAll('[data-i18n]').forEach(el=>{el.innerHTML=texts[lang][el.dataset.i18n]||texts.zh[el.dataset.i18n];});
 const button=document.getElementById('language');button.innerHTML=lang==='zh'?'EN <span>↗</span>':'中文 <span>↗</span>';button.setAttribute('aria-label',lang==='zh'?'Switch to English':'切换为中文');
 renderGallery();updateMetrics();if(document.getElementById('lightbox').open)fillLightbox();
 try{localStorage.setItem('cohere-language',lang);}catch{}
}
function renderGallery(){
 const visible=figures.filter(f=>filter==='all'||f.category===filter);
 document.getElementById('gallery').innerHTML=visible.map(f=>`<article class="gallery-card"><button class="gallery-image figure-trigger" data-figure="${f.id}" aria-label="${lang==='zh'?'放大':'Enlarge'} ${f[lang]}"><img src="assets/figure-${pad(f.id)}.webp" alt="${f[lang]}" loading="lazy" width="1200" height="600"><span class="enlarge" aria-hidden="true">↗</span></button><div class="gallery-card-info"><div><h3>${f[lang]}</h3><p>FIG. ${pad(f.paperFigure)} / ${f.category==='method'?'ARCHITECTURE':f.category.toUpperCase()}</p></div><a href="${original(f.id)}" target="_blank" rel="noopener" aria-label="${lang==='zh'?'打开原始图片':'Open original figure'} ${pad(f.paperFigure)}">PNG ↗</a></div></article>`).join('');
 document.getElementById('figure-count').textContent=`${pad(visible.length)} ${lang==='zh'?'张图示':'FIGURES'}`;
}
function escapeHtml(value){return String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));}
function updateMetrics(){
 if(!comparison)return;
 const selected=comparison.datasets.find(item=>item.key===dataset);
 document.getElementById('dataset-name').textContent=selected.name;
 const rows=[];let previousGroup=null;
 for(const method of comparison.methods){
  if(method.text_input!==previousGroup){
   const label=method.text_input?(lang==='zh'?'需要文本输入':'With text input'):(lang==='zh'?'无需文本输入':'Without text input');
   rows.push(`<tr class="method-group"><th colspan="8" scope="rowgroup">${label}</th></tr>`);previousGroup=method.text_input;
  }
  const own=method.name==='CohereFusion';
  const cells=comparison.metrics.map(metric=>{
   const value=method.results[dataset][metric].toFixed(3),rank=method.ranks[dataset][metric];
   const rankClass=rank==='best'?' rank-best':rank==='second'?' rank-second':'';
   const rankLabel=rank==='best'?(lang==='zh'?'，最优':', best'):rank==='second'?(lang==='zh'?'，次优':', second best'):'';
   return `<td><span class="metric-value${rankClass}" aria-label="${metric} ${value}${rankLabel}">${value}</span></td>`;
  }).join('');
  rows.push(`<tr${own?' class="cohere-row"':''}><th scope="row">${escapeHtml(method.name)}</th><td class="venue-cell">${escapeHtml(method.venue)}</td>${cells}</tr>`);
 }
 document.getElementById('metrics-body').innerHTML=rows.join('');
}
function fillLightbox(){
 const figure=figures.find(f=>f.id===currentFigure);
 document.getElementById('lightbox-title').textContent=figure[lang];
 document.getElementById('lightbox-number').textContent=`FIGURE ${pad(figure.paperFigure)}`;
 const img=document.getElementById('lightbox-img');img.src=`assets/figure-${pad(figure.id)}.webp`;img.alt=figure[lang];
 document.getElementById('original-link').href=original(figure.id);
}
function openFigure(id){currentFigure=id;fillLightbox();document.getElementById('lightbox').showModal();}
function navigateFigure(delta){const visible=figures.filter(f=>filter==='all'||f.category===filter);const index=visible.findIndex(f=>f.id===currentFigure);currentFigure=visible[(index+delta+visible.length)%visible.length].id;fillLightbox();}
document.getElementById('language').addEventListener('click',()=>{lang=lang==='zh'?'en':'zh';setLanguage();});
document.addEventListener('click',event=>{
 const trigger=event.target.closest('.figure-trigger');if(trigger)openFigure(Number(trigger.dataset.figure));
 const filterButton=event.target.closest('[data-filter]');if(filterButton){filter=filterButton.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>{b.classList.toggle('active',b===filterButton);b.setAttribute('aria-pressed',String(b===filterButton));});renderGallery();}
 const datasetButton=event.target.closest('[data-dataset]');if(datasetButton){dataset=datasetButton.dataset.dataset;document.querySelectorAll('[data-dataset]').forEach(b=>{b.classList.toggle('active',b===datasetButton);b.setAttribute('aria-pressed',String(b===datasetButton));});updateMetrics();}
 const commandButton=event.target.closest('[data-command]');if(commandButton){command=commandButton.dataset.command;document.querySelectorAll('[data-command]').forEach(b=>{b.classList.toggle('active',b===commandButton);b.setAttribute('aria-pressed',String(b===commandButton));});document.getElementById('command-code').textContent=commands[command];}
});
document.getElementById('close-lightbox').addEventListener('click',()=>document.getElementById('lightbox').close());
document.getElementById('previous-figure').addEventListener('click',()=>navigateFigure(-1));
document.getElementById('next-figure').addEventListener('click',()=>navigateFigure(1));
document.getElementById('lightbox').addEventListener('click',event=>{if(event.target===event.currentTarget){const rect=event.currentTarget.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)event.currentTarget.close();}});
document.addEventListener('keydown',event=>{if(!document.getElementById('lightbox').open)return;if(event.key==='ArrowLeft'){event.preventDefault();navigateFigure(-1);}if(event.key==='ArrowRight'){event.preventDefault();navigateFigure(1);}});
document.getElementById('copy-command').addEventListener('click',async()=>{
 const status=document.getElementById('copy-status');
 try{await navigator.clipboard.writeText(commands[command]);status.textContent=lang==='zh'?'命令已复制':'Command copied';const button=document.getElementById('copy-command');button.textContent=lang==='zh'?'已复制 ✓':'Copied ✓';setTimeout(()=>{button.textContent=(lang==='zh'?'复制命令':'Copy command')+' ⧉';},1800);}
 catch{status.textContent=lang==='zh'?'请选中命令手动复制':'Select the command to copy it manually';const range=document.createRange();range.selectNodeContents(document.getElementById('command-code'));window.getSelection().removeAllRanges();window.getSelection().addRange(range);}
});
const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting)document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.hash==='#'+entry.target.id));});},{rootMargin:'-15% 0px -60% 0px'});
document.querySelectorAll('main>section[id]').forEach(s=>observer.observe(s));
document.getElementById('command-code').textContent=commands.setup;
setLanguage();
fetch('assets/cross-dataset-results.json').then(response=>{if(!response.ok)throw new Error('Table III data could not be loaded');return response.json();}).then(data=>{comparison=data;updateMetrics();}).catch(()=>{document.getElementById('metrics-body').innerHTML=`<tr><td colspan="8">${lang==='zh'?'表格加载失败，请通过 JSON 链接查看。':'Table unavailable. Open the JSON link to view the source.'}</td></tr>`;});
