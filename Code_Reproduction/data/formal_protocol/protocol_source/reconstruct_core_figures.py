"""Extend the original vector icon language to the three explanatory figures."""
from pathlib import Path
from xml.etree import ElementTree as E
import copy,io,sys,math,json,hashlib,re
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
matplotlib.rcParams.update({'mathtext.fontset':'stix','svg.fonttype':'path'})

BASE=Path(__file__).resolve().parents[1];FIG=BASE/'Paper_Source/figures'
sys.path.insert(0,str(BASE.parents[1]/'work/figure_deps'))
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
for n,f in [('Times New Roman','times.ttf'),('Times New Roman Bold','timesbd.ttf')]:
    pdfmetrics.registerFont(TTFont(n,'C:/Windows/Fonts/'+f))
NS='http://www.w3.org/2000/svg';E.register_namespace('',NS);E.register_namespace('xlink','http://www.w3.org/1999/xlink')
original=E.parse(FIG/'architecture_original.svg').getroot();old=list(original)
NAVY='#244B6B';TEAL='#2E6258';GOLD='#8C7145';GRAY='#667078';RED='#A94B45';INK='#202B35'
root=None;serial=0

def el(tag,parent=None,**kw):
    return E.SubElement(root if parent is None else parent,'{'+NS+'}'+tag,{k.replace('_','-'):str(v) for k,v in kw.items()})
def start(w,h):
    global root,serial
    serial=0;root=E.Element('{'+NS+'}svg',dict(width=str(w),height=str(h),viewBox=f'0 0 {w} {h}'))
    root.append(copy.deepcopy(old[0]));root.append(copy.deepcopy(old[55]));el('rect',width=w,height=h,fill='white')
    el('style').text='text{font-family:"Times New Roman",Times,serif}'
def txt(x,y,s,size=23,color=INK,bold=False,anchor='start'):
    e=el('text',x=x,y=y,font_size=size,fill=color,font_weight='bold' if bold else 'normal',text_anchor=anchor);e.text=s;return e
def line(points,color=NAVY,width=2.3,arrow=False,dash=None):
    e=el('polyline',points=' '.join(f'{x},{y}' for x,y in points),fill='none',stroke=color,stroke_width=width,stroke_linejoin='round')
    if dash:e.set('stroke-dasharray',dash)
    if arrow:
        (a,b),(x,y)=points[-2:];length=math.hypot(x-a,y-b);dx=(x-a)/length;dy=(y-b)/length
        el('polygon',points=f'{x},{y} {x-10*dx+4*dy},{y-10*dy-4*dx} {x-10*dx-4*dy},{y-10*dy+4*dx}',fill=color)
def eq(expr,x,y,width=400,size=24,color=INK):
    global serial
    for command in ('mathcal','overline','bar','mathfrak','mathbf'):
        expr=re.sub(r'\\'+command+r'\s*([A-Za-z0-9])',lambda m:'\\'+command+'{'+m.group(1)+'}',expr)
    expr=expr.replace(r'\mathbf1',r'\mathbf{1}').replace(r'\not\to',r'\nrightarrow').replace(r'\text{feedback}',r'\mathrm{feedback}')
    serial+=1;f=Figure(figsize=(11,.5));f.text(0,0,'$'+expr+'$',fontsize=12,color=color)
    b=io.BytesIO();f.savefig(b,format='svg',bbox_inches='tight',pad_inches=.02,transparent=True)
    fragment=E.fromstring(b.getvalue());natural=float(fragment.get('width').replace('pt',''));scale=min(size/12,width/natural)
    prefix='eq'+str(serial)+'_'
    for e in fragment.iter():
        if e.get('id'):e.set('id',prefix+e.get('id'))
        for k,v in list(e.attrib.items()):
            if k.endswith('href') and v.startswith('#'):e.set(k,'#'+prefix+v[1:])
            elif 'url(#' in v:e.set(k,v.replace('url(#','url(#'+prefix))
    g=el('g',transform=f'translate({x},{y}) scale({scale})')
    for e in fragment:
        if not e.tag.endswith('metadata'):g.append(e)
def worker(x,y,label,color=NAVY,scale=.63):
    # Same drawn mobile-worker icon as the original architecture, not a bitmap.
    g=copy.deepcopy(old[74]);g.set('transform',f'translate({x},{y}) scale({scale})');root.append(g)
    txt(x+18,y+92*scale,label,20,color,True,'middle')
def token(x,y,label,color=TEAL):
    el('rect',x=x,y=y,width=59,height=69,rx=3,fill='#F5F9F8',stroke=color,stroke_width=1.8)
    for v in (16,26,36):line([(x+10,y+v),(x+44,y+v)],color,1.4)
    el('circle',cx=x+47,cy=y+59,r=11,fill='white',stroke=color,stroke_width=1.6)
    line([(x+40,y+59),(x+45,y+64),(x+54,y+53)],color,2.0)
    txt(x+29,y+95,label,19,color,False,'middle')
def grid(x,y,rows,cols,filled,p=19):
    for a in range(rows):
        for b in range(cols):
            z=filled.get((a,b),0);colors=['#F6F8F9','#426F64','#A5C3B5','#8EAAC0']
            el('rect',x=x+b*p,y=y+a*p,width=p-2,height=p-2,fill=colors[z],stroke='#B8C9C5',stroke_width=.6)
def map_panel(x,y,scale=.64):
    m=copy.deepcopy(old[56]);m.set('x',str(x));m.set('y',str(y));m.set('width',str(823*scale));m.set('height',str(390*scale));root.append(m)
def region(x,y,label,color,filled=True):
    el('circle',cx=x,cy=y,r=20,fill='#EFF6F1' if filled else 'white',stroke=color,stroke_width=2)
    txt(x,y+7,label,24,color,True,'middle')
def save(name):
    svg=FIG/(name+'.svg');E.ElementTree(root).write(svg,encoding='utf-8',xml_declaration=True)
    d=svg2rlg(str(svg));renderPDF.drawToFile(d,str(FIG/(name+'.pdf')))

start(1200,440)
txt(20,29,'(a) Sparse references leave workers unknown',25,NAVY,True)
txt(625,29,'(b) Credible anchors expand the checking domain',25,TEAL,True)
line([(600,40),(600,420)],'#B9CAC6',1.1)
for x in (15,620):map_panel(x,53,.67)
for x in (0,605):
    region(x+161,132,'a',TEAL);region(x+326,201,'b',GOLD,False);region(x+468,113,'c',GOLD,False)
    worker(x+103,189,'C',TEAL,.54);worker(x+307,246,'U',GOLD,.54);worker(x+457,174,'U',GOLD,.54)
    line([(x+127,196),(x+148,151)],TEAL,2,True)
    line([(x+325,246),(x+326,223)],GOLD,2,True)
    line([(x+477,177),(x+472,136)],GOLD,2,True)
txt(38,343,'Measured anchor',22,TEAL,True)
txt(277,343,'No direct check',22,GOLD,True)
eq(r'\mathcal{A}_t\subsetneq\mathcal{O}_t',170,367,350,26,NAVY)
line([(784,133),(850,133),(910,190)],TEAL,2.8,True,'7 4')
line([(797,124),(984,124),(1054,119)],TEAL,2.8,True,'7 4')
txt(836,93,'Public-history model',22,TEAL,True)
txt(650,342,'Actual report + supported reference',23,TEAL,True)
eq(r'\mathcal E_i=\mathcal D_i\cup\mathcal H_i,\quad y_i\not\to h',654,369,510,26,TEAL)
save('motivation_reference')

start(1200,430)
txt(18,30,'(a) Graded private offers',25,NAVY,True)
txt(442,30,'(b) Budgeted single-round choice',25,NAVY,True)
txt(893,30,'(c) Adaptive roster',25,TEAL,True)
for i in range(4):
    worker(27+86*i,60,'C' if i<2 else 'U',TEAL if i<2 else GOLD,.5)
    for j in range(5):el('rect',x=20+86*i+j*11,y=153,width=9,height=12,fill=TEAL if j<5-i else '#DEE6E3')
    grid(20+86*i,187,3,4,{(a,b):1 if (a+b+i)%3==0 else 2 for a in range(3) for b in range(4)},15)
eq(r'(\bar r_i,\widetilde b_i,\widetilde c_i)',34,270,326,28,NAVY)
line([(360,184),(427,184)],NAVY,3,True)
eq(r'F(L)=\overline v(L)+\lambda_Q\overline o(L)',443,85,399,28,NAVY)
eq(r'\Delta_i(L)=\frac{[\Delta F]_++\delta_F}{\max\{\Delta\mathfrak q,\delta_q\}}(b_0+b_1\bar r_i)',434,148,410,28,NAVY)
eq(r'Q(L\cup\{i\})\leq B_t',468,238,350,30,TEAL)
line([(845,184),(887,184)],NAVY,3,True)
for i in range(4):worker(917+63*(i%3),67+112*(i//3),'C' if i<2 else 'U',TEAL if i<2 else GOLD,.5)
line([(927,332),(1149,332)],GRAY,1.5,True);line([(927,332),(927,238)],GRAY,1.5,True)
line([(937,255),(976,255),(1080,315),(1119,315)],TEAL,3)
eq(r'u_t',888,250,45,22,TEAL);eq(r'f_C',1107,345,55,22,TEAL)
eq(r'u_{\max}',938,232,90,20,TEAL);eq(r'u_{\min}',1058,275,90,20,TEAL)
txt(25,391,'Same virtual coordinates',22,NAVY)
txt(464,391,'One roster; exact aggregate budget',22,NAVY)
txt(920,391,'Fewer unknown slots as C grows',19,TEAL)
save('recruitment_idea')

start(1200,520)
txt(20,30,'(a) Keep worker reports and references distinct',25,TEAL,True)
txt(666,30,'(b) One bounded, evidence-sensitive update',25,TEAL,True)
for i,lab in enumerate(['C','U','U']):
    worker(22,64+i*85,lab,TEAL if i==0 else GOLD,.43)
    grid(91,71+i*85,1,5,{(0,j):(1 if i==0 else 3) for j in range(5) if (i+j)%3!=1},22)
txt(87,340,'Actual reports',21,NAVY,True)
line([(210,87),(272,87)],TEAL,2.8,True)
grid(283,63,4,6,{(a,b):1 if (a+b)%4 else 2 for a in range(4) for b in range(6)},18)
txt(278,171,'Credible anchors + history',21,TEAL,True)
eq(r'\alpha=(K+\lambda I)^{-1}(y-b\mathbf1)',242,201,187,25,TEAL)
eq(r'\mathbf1^T\alpha=0',287,241,145,25,TEAL)
line([(396,100),(471,100),(471,192)],TEAL,2.6,True)
eq(r'd_{ij}:\ \mathcal C_t\setminus\{i\}',467,207,180,24,TEAL)
eq(r'h_j:\ j\notin\mathcal A_t,\ E_j\geq\zeta_R',446,263,210,24,TEAL)
line([(207,171),(232,171),(232,365),(507,365)],NAVY,2.4,True)
txt(278,354,'Separate service fusion',20,NAVY,True)
eq(r'\widehat x_j\ \not\to\ \text{feedback}',430,384,218,22,RED)
line([(640,224),(665,224),(665,185),(693,185)],TEAL,2.6,True)
grid(709,80,2,6,{(a,b):1 if b<3 else 2 for a in range(2) for b in range(6)},26)
txt(713,158,'Direct',21,TEAL,True);txt(805,158,'Recovered',21,TEAL,True)
eq(r'w_{ij}\in\{0,\beta,1\},\quad N_i=\sum_j w_{ij}',698,189,455,27,TEAL)
eq(r'\overline s_i=\frac{\sum_jw_{ij}s(r_i,e_{ij})}{N_i}',733,240,410,30,TEAL)
eq(r'\Delta r_i=\sqrt{\min(N_i,K_E)}\;\overline s_i',708,309,445,31,TEAL)
line([(930,354),(930,388)],TEAL,2.4,True)
token(870,397,'successor',TEAL);token(967,397,'fresh grade',TEAL)
txt(25,448,'No self-reference; no fabricated worker observation',23,TEAL,True)
save('feedback_idea')

names=['motivation_reference','recruitment_idea','feedback_idea']
(FIG/'core_figure_manifest.json').write_text(json.dumps(dict(source='architecture_original.svg vector map and worker icon; new mechanism equations',
    architecture_source_sha256=hashlib.sha256((FIG/'architecture_original.svg').read_bytes()).hexdigest(),
    figures={n:{'role':r,'svg_sha256':hashlib.sha256((FIG/(n+'.svg')).read_bytes()).hexdigest()} for n,r in zip(names,
      ['motivate sparse reference bottleneck','explain signed-grade single-round recruitment','separate reference construction from bounded feedback'])}),indent=2),encoding='utf-8')
print('Rendered three original-style vector explanations')
