"""Reconstruct the supplied artwork in place, preserving its map and icon paths."""
import copy,io,sys,hashlib,json,re,math
from pathlib import Path
from xml.etree import ElementTree as E
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
matplotlib.rcParams.update({'mathtext.fontset':'stix','svg.fonttype':'path'})

BASE=Path(__file__).resolve().parents[1]
FIG=BASE/'Paper_Source'/'figures'
NS='http://www.w3.org/2000/svg'
E.register_namespace('',NS);E.register_namespace('xlink','http://www.w3.org/1999/xlink')
source=FIG/'architecture_original.svg';root=E.parse(source).getroot();old=list(root)
NAVY='#244B6B';TEAL='#2E6258';GOLD='#8C7145';GRAY='#444B53';INK='#202B35'

# Keep the original sensing map, enrollment building, worker figures,
# randomized-response gates, phone, encrypted envelopes and blockchain icons.
remove={23,53,76,77,112,113,114,115,116,142,143,144,145,146,178,201,204,247,248,249,250,251,252,253,254,255,256,257,278,
        308,312,314,316,321,322}|set(range(325,len(old)))
for i in sorted(remove):root.remove(old[i])
replacements={
 177:'Signed RR offer + signed reputation grade',
 181:'Blockchain platform B',198:'Bound inputs',203:'Round binding',
 208:'Old state',216:'Locked t',224:'Next state',
 240:'Roles / epoch',243:'Cap / roster',246:'State / replay',
 277:'Selected-worker receipt',309:'Actual readings + region mask; fresh alias signature',
 315:'Requester-only ciphertexts',317:'Locked roster',
 320:'State renewal',323:'Open / Lock / Collect / Settle',
 324:'Roles, bindings, cap and replay checks',
}
for i,s in replacements.items():old[i].text=s
old[309].set('font-size','24');old[315].set('font-size','23')
old[323].set('font-size','24');old[324].set('font-size','23')
old[19].set('width','690');old[20].set('width','688');old[21].set('width','688')
old[22].set('d','M20,1130 L710,1130')
old[319].text='6';old[320].set('font-size','28')

def node(tag,parent=None,**kw):
    return E.SubElement(root if parent is None else parent,'{'+NS+'}'+tag,{k.replace('_','-'):str(v) for k,v in kw.items()})
def text(x,y,s,size=25,color=INK,bold=False,center=False):
    e=node('text',x=x,y=y,font_size=size,fill=color,font_weight='bold' if bold else 'normal',text_anchor='middle' if center else 'start');e.text=s;return e
def rect(x,y,w,h,fill,stroke='none',r=0,lw=1):
    return node('rect',x=x,y=y,width=w,height=h,rx=r,fill=fill,stroke=stroke,stroke_width=lw)
def path(d,color=NAVY,arrow=False,dashed=False,lw=3.0):
    kw=dict(d=d,fill='none',stroke=color,stroke_width=lw,stroke_linejoin='round',stroke_linecap='round')
    if dashed:kw['stroke_dasharray']='7 5'
    result=node('path',**kw)
    if arrow:
        # Explicit heads survive SVG-to-PDF converters that omit SVG markers.
        point=(0.,0.);previous=None
        for command,args in re.findall(r'([MLHVC])([^MLHVC]*)',d):
            v=[float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',args)]
            if command in ('M','L'):
                for k in range(0,len(v),2):previous,point=point,(v[k],v[k+1])
            elif command=='H':previous,point=point,(v[-1],point[1])
            elif command=='V':previous,point=point,(point[0],v[-1])
            elif command=='C':previous,point=(v[-4],v[-3]),(v[-2],v[-1])
        if previous:
            dx,dy=point[0]-previous[0],point[1]-previous[1];length=math.hypot(dx,dy)
            if length:
                ux,uy=dx/length,dy/length;x,y=point
                pts=[(x,y),(x-10*ux+4.5*uy,y-10*uy-4.5*ux),(x-10*ux-4.5*uy,y-10*uy+4.5*ux)]
                node('polygon',points=' '.join(f'{a},{b}' for a,b in pts),fill=color)
    return result
eqnum=0
def eq(expr,x,y,maxwidth,size=26,color=INK,center=False):
    global eqnum
    eqnum+=1
    fig=Figure(figsize=(12,.5));fig.text(0,0,'$'+expr+'$',fontsize=12,color=color)
    b=io.BytesIO();fig.savefig(b,format='svg',bbox_inches='tight',pad_inches=.02,transparent=True)
    frag=E.fromstring(b.getvalue());width=float(frag.get('width').replace('pt',''))
    scale=min(size/12,maxwidth/width);prefix='new_eq_'+str(eqnum)+'_'
    if center:x-=width*scale/2
    for e in frag.iter():
        if e.get('id'):e.set('id',prefix+e.get('id'))
        for k,v in list(e.attrib.items()):
            if k.endswith('href') and v.startswith('#'):e.set(k,'#'+prefix+v[1:])
            elif 'url(#' in v:e.set(k,v.replace('url(#','url(#'+prefix))
    g=node('g',transform=f'translate({x},{y}) scale({scale})')
    for e in list(frag):
        if not e.tag.endswith('metadata'):g.append(e)
def step(n,x,y,title,color=TEAL,size=28):
    node('circle',cx=x,cy=y-9,r=21,fill='white',stroke=color,stroke_width=2)
    text(x,y,str(n),29,color,True,True);text(x+37,y,title,size,color,True)
def matrix(x,y,vals,pitch=23):
    cols=['#F8FBFA','#4C7F70','#A5C2B7','#6B8E9D']
    for j,row in enumerate(vals):
        for i,v in enumerate(row):
            rect(x+i*pitch,y+j*pitch,pitch-3,pitch-3,cols[v],'#A8C3B8',1,.6)
            if v==0:path(f'M{x+i*pitch+3},{y+j*pitch+pitch-6} l{pitch-10},-{pitch-10}','#C3D6CE',False,False,.7)
def credential(x,y,label,color):
    node('circle',cx=x,cy=y,r=24,fill='#F5FAF7',stroke=color,stroke_width=1.8)
    node('circle',cx=x,cy=y-7,r=6,fill=color)
    path(f'M{x-12},{y+12} q12,-24 24,0',color,False,False,2)
    text(x,y+55,label,22,TEAL,True,True)

# Replace the obsolete weight commitment with the actual signed state.
eq(r'W_1',129,340,55,26,NAVY);eq(r'W_2',706,475,55,26,NAVY)
# Restore explicit arrowheads on the inherited protocol connectors.
for index in (54,80,81,103,106,117,133,136,147,153,155,179,280,313):
    e=old[index]
    if e in list(root):root.remove(e)
    path(e.get('d'),e.get('stroke',NAVY),True,bool(e.get('stroke-dasharray')),3.0)
e=old[151];root.remove(e);path('M1644,268 V460',NAVY,True,True,2.8)
# Direction markers sit between route samples, not on their numbered icons.
for d in ('M254,408 L284,395','M341,429 L352,455','M421,480 L447,469',
          'M648,476 L637,450','M657,380 L685,369','M758,385 L768,408'):
    path(d,NAVY,True,False,3.8)
for y,indices in [(198,[112,113,114,115]),(383,[142,143,144,145])]:
    for index,expr in zip(indices,['p','1-p','1-p','p']):
        original=old[index]
        eq(expr,float(original.get('x')),float(original.get('y'))-19,35,22,NAVY,True)
eq(r'\epsilon_1',1321,253,60,25,NAVY);eq(r'\epsilon_2',1321,438,60,25,NAVY)
eq(r'\mathrm{cert}_A',208,174,120,26,GOLD)
eq(r'\tau_{i,t}=\mathrm{Sign}_R(t,\mathrm{sid}_{i,t},\bar r_{i,t},H(\mathrm{tok}_{i,t}))',1040,582,510,25,NAVY)
text(1370,680,'Broker S: validate + route',25,NAVY)
for i in (247,248,249,250,251):
    # The original laptop hardware is retained at its original position.
    root.append(old[i])
path('M1583,641 H1343',NAVY,True,True)
# Offers reach the broker selection lane rather than a proof-planning circuit.
path('M1269,690 V703 H9 V1076 H360 V1085',NAVY,True)
eq(r'(t,\rho_t,\mathrm{policy},B_t)',1162,770,367,25)
eq(r'\rho_t',797,1010,75,25)
eq(r'L_t',158,826,90,28,NAVY)
eq(r'\sigma_i=\mathrm{Sign}_{a_{i,t}}(t,H(\mathrm{ct}_{i,t}))',155,923,410,25,NAVY)
eq(r'\mathrm{Verify}_{a_{i,t}}(\sigma_i)=1',525,923,182,21,NAVY)
# Settled token batches, not numerical-correctness proofs.
eq(r'C_i^\prime=C_i s_i^{n_P},\quad\mathrm{sid}_{i,t+1}\leftarrow\{0,1\}^{\lambda_{\rm sid}}',1009,973,490,24,TEAL)
eq(r'\rho_t\longrightarrow\rho_{t+1}',1480,1030,270,28)

# Split the original lower processing panel according to the new entity roles.
rect(730,1086,1050,456,'#DDE9E4',TEAL,12,2.4)
rect(731,1087,1048,44,'#C6D8D0','none',11)
rect(731,1115,1048,15,'#C6D8D0')
path('M730,1130 H1780',TEAL,False,False,1)
text(44,1118,'Broker S · single-round selection',27,TEAL,True)
text(756,1118,'Requester R · post-collection recovery and scoring',27,TEAL,True)

step(3,51,1176,'Reputation-graded recruitment',TEAL,27)
credential(105,1227,'Credible',TEAL);credential(244,1227,'Unknown',GOLD)
path('M135,1227 H211',TEAL,True)
eq(r'u_t\in[u_{\min},u_{\max}],\quad u_t\downarrow\ \mathrm{as}\ |C_t|\uparrow',330,1207,340,26,TEAL)
text(70,1317,'Signed grades + perturbed virtual coverage',24,TEAL,True)
matrix(80,1343,[[1,0,2],[0,1,0],[2,0,1]],23)
path('M160,1374 H198',TEAL,True)
eq(r'F(L)=\overline{v_j(L)}+\frac{\lambda_Q}{n}\sum_{i\in U_L}o_i(L)(a_0+a_1z_i)',212,1344,461,25,TEAL)
eq(r'i^*=\arg\max_i\frac{(\Delta F_i+\delta_F)(b_0+b_1\bar r_i)}{\max\{\Delta\mathfrak{q}_i,\delta_q\}}',212,1392,461,25,TEAL)
eq(r'Q(L\cup\{i\})\leq B_t,\quad\bar r_i\in\Gamma_g',70,1455,597,26,TEAL)
text(70,1518,'Prefer C/U; fall back only if absent or unaffordable',23,TEAL)
# A broker decision locks the selected roster before a report is collected.
path('M360,1085 V1077 H745 V811 H778',TEAL,True)
path('M778,811 H652 V777 H365',GRAY,True)

step(5,766,1176,'Credible anchors / secondary references / bounded update',TEAL,26)
matrix(775,1206,[[3,0,0,0],[0,1,0,3],[0,0,1,0],[0,3,0,0]],22)
eq(r'y^C_{\mathcal{A}_t}',781,1302,108,25,TEAL)
path('M872,1243 H912',TEAL,True)
matrix(932,1215,[[1,1,2,2],[1,1,1,2],[2,1,1,2],[2,2,2,1]],18)
text(931,1325,'Kernel fit',21,TEAL,True)
eq(r'h_j=b_j+\mathbf{k}(y^C_{\mathcal{A}_t})^\top\alpha_j',1053,1204,455,26,TEAL)
eq(r'\mathcal{H}_t=\{j\notin\mathcal{A}_t:\mathrm{EV}_j\geq\zeta_R\}',1053,1251,455,26,TEAL)
text(1053,1318,'Actual U report; no direct C reference',22,TEAL)
path('M1523,1258 H1573',TEAL,True)
text(1547,1292,'Decode',17,TEAL,False,True)
matrix(1582,1206,[[3,2,0,1],[2,1,2,3],[0,2,1,2],[3,2,0,1]],29)
text(1638,1191,'Task service',22,TEAL,True,True)
eq(r'\widehat x_j=a+\Delta_g\widehat g_j',1530,1328,217,21,TEAL)
# Region evidence combines once, after all references have been frozen.
eq(r'd_{ij}=\mathrm{TD}(C_j\setminus\{i\}),\quad N_i=|D_i|+\beta|H_i|',773,1375,565,26,TEAL)
eq(r'r_i^+=Q_h\!\left[\Pi_{[0,1]}\!\left(r_i+\sqrt{\min(N_i,K_E)}\,\bar s_i\right)\right]',773,1430,565,28,TEAL)
text(773,1518,'Frozen references; no self-evaluation',23,TEAL)
# Reuse the original encrypted-document icon paths as a successor-token stack.
text(1460,1375,'Encrypted successors',20,TEAL,True,True)
for dx,dy in [(0,0),(18,-14),(36,-28)]:
    icon=node('g',transform=f'translate({1370+dx},{1420+dy}) scale(1.2) translate(-204,-847)')
    for index in range(281,290):icon.append(copy.deepcopy(old[index]))
path('M1534,1442 C1559,1416 1595,1416 1621,1442',TEAL,True)
path('M1534,1458 C1559,1484 1595,1484 1621,1458',TEAL,True)
icon=node('g',transform='translate(1640,1420) scale(1.15) translate(-204,-847)')
for index in range(281,290):icon.append(copy.deepcopy(old[index]))
path('M1735,1455 l10,12 l18,-28',TEAL,False,False,3)
eq(r'\tau_i^+=\mathrm{Sign}_R(\mathrm{hdr}_i^+)',1370,1493,370,23,TEAL)
text(1556,1536,'Digest / serial / grade; ciphertext stays at S',17,TEAL,False,True)
# Vertical interface uses only the gap between the original ledger and requester.
path('M1744,1085 V1077 H1772 V1069',TEAL,True)
# Ledger blocks advance left-to-right in addition to their chain-link icons.
path('M1173,874 H1192',GRAY,True,False,2.8)
path('M1368,874 H1387',GRAY,True,False,2.8)
# Header certificates return to the broker for full-token assembly.
path('M730,1418 H686',TEAL,True)
eq(r'\sigma_t^+',678,1425,47,22,TEAL)
# The report cable enters the requester through the side port. Its white
# under-stroke marks a non-junction where it crosses the selection cable.
report_path='M596,876 H675 V1068 H720 V1243 H770'
node('path',d=report_path,fill='none',stroke='white',stroke_width=7.5,stroke_linejoin='round')
path(report_path,GRAY,True,False,3.0)
text(1400,1100,'',22,TEAL)
metadata=node('metadata');metadata.text=json.dumps(dict(base='architecture_original.svg',base_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    kept_original_root_elements=len(old)-len(remove),new_mechanism='signed-grade/current-round/credible-only secondary/bounded update'),ensure_ascii=False)
dest=FIG/'requester_architecture.svg';E.ElementTree(root).write(dest,encoding='utf-8',xml_declaration=True)

# Convert the preserved SVG paths to a vector PDF with the original serif face.
sys.path.insert(0,str(BASE.parent/'figure_deps'))
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
for name,file in [('Times New Roman','times.ttf'),('Times New Roman-Bold','timesbd.ttf'),('Times New Roman-Italic','timesi.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Times New Roman',normal='Times New Roman',bold='Times New Roman-Bold',italic='Times New Roman-Italic',boldItalic='Times New Roman-Bold')
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
drawing=svg2rlg(str(dest));renderPDF.drawToFile(drawing,str(FIG/'requester_architecture.pdf'))
print(json.dumps(dict(original_elements_retained=len(old)-len(remove),vector_pdf=True)))
