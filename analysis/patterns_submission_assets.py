#!/usr/bin/env python3
"""Rebuild submission tables/figures and supplementary checks from archived data.

Run patterns_statistics.py first. No model API calls or private human exports.
Bootstrap intervals condition on recorded responses, resampling base-problem IDs
within stimulus families; positive/negative queries of a base stay together.
"""
from pathlib import Path
import json
import hashlib
import platform
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import patterns_statistics as ps

OUT=ps.OUT
GEN=ps.ROOT/'paper'/'generated'
FIG=ps.PAPER_FIGURES
SEED=20260907
NBOOT=10000
NAMES={'gemini-3.5-flash':'Gemini 3.5 Flash','gpt-5.5':'GPT 5.5','qwen3-vl:30b':'Qwen3-VL:30b','gemma4:31b':'Gemma4:31b'}

def display(model):
    """Publication name for a model alias.

    NAMES covers the aliases that are not simply capitalised. Anything else is
    derived, so a newly qualified model formats sensibly instead of raising:
    the top-ups promoted gemma4:26b into the paired comparison after that map
    was written, and the table builder was the one place still indexing it
    directly.
    """
    if model in NAMES:
        return NAMES[model]
    vendor, _, tag = str(model).partition(':')
    vendor = (vendor[:1].upper() + vendor[1:]).replace('-vl', '-VL')
    return f'{vendor}:{tag}' if tag else vendor
FAMILIES={'bd':'Basic','ff':'Free-form','hd_comb':'Human-defined','hd_novel':'Human-defined Novel'}
FAMILY_BY_QUERY = None

def tex(s):
    return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')

def ptex(p):
    return '$<0.001$' if p<.001 else f'{p:.3f}'

def savefig(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    plt.close(fig)

def decorate(frame,idcol):
    f=frame.copy()
    f['base']=f[idcol].str.replace(r'_(pos|neg)$','',regex=True)
    global FAMILY_BY_QUERY
    if FAMILY_BY_QUERY is None:
        FAMILY_BY_QUERY = (pd.read_csv(ps.HUMAN, usecols=['question_id', 'category'])
                           .drop_duplicates('question_id').set_index('question_id').category.to_dict())
    f['family']=f[idcol].map(FAMILY_BY_QUERY)
    assert f['family'].notna().all(), 'Query missing from human inventory'
    return f

def cluster_indices(frame,rng):
    # Stable family-stratified clusters, including all queries of each base.
    blocks=[]
    for _,g in frame.groupby('family',sort=True):
        groups=[x.index.to_numpy() for _,x in g.groupby('base',sort=True)]
        blocks.extend(groups[j] for j in rng.integers(0,len(groups),len(groups)))
    return np.concatenate(blocks)

def paired_sdt(machine):
    rows=[]
    no=machine[machine.concept_flag.eq('no_concept')]
    rng=np.random.default_rng(SEED)
    for model,g in no.groupby('model'):
        l=g[g.representation.eq('DVRL')].set_index('test_id')
        r=g[g.representation.eq('AD')].set_index('test_id')
        ids=l.index.intersection(r.index)
        if len(ids)<40: continue
        f=decorate(pd.DataFrame({'test_id':ids,'truth':l.loc[ids,'truth_positive'].to_numpy(),
              'left':l.loc[ids,'response_positive'].to_numpy(), 'right':r.loc[ids,'response_positive'].to_numpy()}),'test_id')
        # Vectorized cluster bootstrap of signal and response counts.
        t=f.truth.to_numpy(); a=f.left.to_numpy(); b=f.right.to_numpy()
        counts=pd.DataFrame({'base':f.base,'family':f.family,'np':t.astype(int),'nn':(~t).astype(int),
                 'hl':(t&a).astype(int),'fl':((~t)&a).astype(int),'hr':(t&b).astype(int),'fr':((~t)&b).astype(int),
                 'cl':(t==a).astype(int),'cr':(t==b).astype(int)}).groupby(['family','base']).sum()
        boot=np.zeros((NBOOT,8))
        for _,block in counts.groupby(level=0):
            weights=rng.multinomial(len(block),np.full(len(block),1/len(block)),size=NBOOT)
            boot+=weights@block.to_numpy()
        def metrics(v):
            npos,nneg,hl,fl,hr,fr,cl,cr=np.asarray(v).T
            zl=norm.ppf((hl+.5)/(npos+1)); zfl=norm.ppf((fl+.5)/(nneg+1))
            zr=norm.ppf((hr+.5)/(npos+1)); zfr=norm.ppf((fr+.5)/(nneg+1))
            return np.array([zl-zfl,zr-zfr,-(zl+zfl)/2,-(zr+zfr)/2,(cr-cl)/(npos+nneg)*100]).T
        point=metrics(counts.sum().to_numpy())
        bm=metrics(boot)
        dd=bm[:,1]-bm[:,0]; dc=bm[:,3]-bm[:,2]
        rows.append({'model':model,'n_items':len(f),'n_base_problems':f.base.nunique(),
            'n_positive':int(t.sum()),'n_negative':int((~t).sum()),
            'pixel_dprime':point[0],'description_dprime':point[1], 'delta_dprime':point[1]-point[0],
            'delta_dprime_low':np.quantile(dd,.025),'delta_dprime_high':np.quantile(dd,.975),
            'pixel_criterion':point[2],'description_criterion':point[3],'delta_criterion':point[3]-point[2],
            'delta_criterion_low':np.quantile(dc,.025),'delta_criterion_high':np.quantile(dc,.975),
            'delta_accuracy_pp':point[4],'delta_accuracy_low':np.quantile(bm[:,4],.025),'delta_accuracy_high':np.quantile(bm[:,4],.975)})
    result=pd.DataFrame(rows); result.to_csv(OUT/'machine_paired_sdt.csv',index=False)
    return result

def alignment_table(h,m):
    def match(hh,mm):
        a=hh.groupby('question_id').correct.mean().rename('human')
        b=mm.groupby('test_id').correct.mean().rename('machine')
        f=pd.concat([a,b],axis=1).dropna().rename_axis('test_id').reset_index()
        f['positive']=~f.test_id.str.endswith('_neg')
        return decorate(f,'test_id')
    def corr(f):
        out=[]
        for sub in [f,f[f.positive],f[~f.positive]]:
            r,p=spearmanr(sub.human,sub.machine)
            out.append((float(r),float(p),len(sub)))
        return out
    primary=match(h,m[m.representation.eq('AD')])
    primary.to_csv(OUT/'alignment_items.csv',index=False)
    point=corr(primary); rng=np.random.default_rng(SEED+1)
    boot=[]
    groups_by_family = [
        [block.index.to_numpy() for _, block in family.groupby('base', sort=True)]
        for _, family in primary.groupby('family', sort=True)
    ]
    for _ in range(NBOOT):
        indices = np.concatenate([
            block[j] for block in groups_by_family
            for j in rng.integers(0, len(block), len(block))
        ])
        sampled=primary.loc[indices]
        vals=corr(sampled)
        boot.append([v[0] for v in vals])
    boot=np.array(boot)
    summary={k:{'r':r,'p':p,'n':n,'ci95':np.quantile(boot[:,i],[.025,.975]).tolist()}
             for i,(k,(r,p,n)) in enumerate(zip(['overall','positive','negative'],point))}
    summary['positive_minus_negative']={'difference':point[1][0]-point[2][0],
                      'ci95':np.quantile(boot[:,1]-boot[:,2],[.025,.975]).tolist()}
    scenarios=[]
    for rep in ['AD','DVRL']:
        for cue in [0,1]:
            f=match(h[h.cue.eq(cue)],m[m.representation.eq(rep)&m.concept_flag.eq('concept' if cue else 'no_concept')])
            vals=corr(f)
            scenarios.append({'comparison':f'{rep}, '+('cue' if cue else 'no cue'),**{k:v[0] for k,v in zip(['overall','positive','negative'],vals)},'n_items':len(f)})
    ad=m[m.representation.eq('AD')]
    counts=ad.groupby(['model','concept_flag']).test_id.nunique().unstack(fill_value=0)
    panel=counts.index[(counts.min(axis=1)>=190)].tolist()
    balanced=ad[ad.model.isin(panel)]
    # Require every model x cue cell to contribute, so composition is constant.
    n=balanced.groupby('test_id').size(); complete=n[n.eq(2*len(panel))].index
    vals=corr(match(h,balanced[balanced.test_id.isin(complete)]))
    scenarios.append({'comparison':'AD, fixed model panel (pooled cues)',**{k:v[0] for k,v in zip(['overall','positive','negative'],vals)},'n_items':int(len(complete))})
    summary['fixed_panel_models']=panel
    pd.DataFrame(scenarios).to_csv(OUT/'alignment_sensitivity.csv',index=False)
    summary['sensitivity']=scenarios
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.8),sharex=True,sharey=True)
    for ax,pol,k in zip(axes,[True,False],['positive','negative']):
        f=primary[primary.positive.eq(pol)]
        ax.scatter(f.human*100,f.machine*100,c='#287B8E' if pol else '#D07835',s=22,alpha=.65,edgecolors='none')
        v=summary[k]; lo,hi=v['ci95']
        ax.set_title(f"{'Positive' if pol else 'Negative'} queries (n = {len(f)})\n"+f"$r_s$ = {v['r']:.3f}; 95% CI [{lo:.2f}, {hi:.2f}]",fontsize=10)
        ax.set(xlim=(-3,103),ylim=(-3,103),xlabel='Human accuracy (%)')
        ax.grid(alpha=.15); ax.set_axisbelow(True)
    axes[0].set_ylabel('Model description accuracy (%)')
    fig.tight_layout();savefig(fig,'patterns_alignment')
    return summary

def coverage(raw):
    rows=[]
    for (model,rep,cue),g in raw.groupby(['model','representation','concept_flag']):
        valid=g.prediction_valid.astype(str).str.lower().eq('true')
        keep=g.clean_primary_keep.astype(str).str.lower().eq('true')
        rows.append({'model':model,'representation':rep,'cue':cue,'archived_rows':len(g),'valid':int(valid.sum()),
          'unique_valid_items':g.loc[valid,'test_id'].nunique(),'families':g.loc[valid,'dataset_family'].nunique(),
          'failure_flagged':int(g.invalid_response_flag.sum()),
          'positive_valid':int((g.loc[valid,'polarity_label']=='Positive').sum()),
          'negative_valid':int((g.loc[valid,'polarity_label']=='Negative').sum()),
          'retained':bool(keep.any()),'source_files':'; '.join(sorted(g.source_file.dropna().unique())),
          'provider_snapshot':'not recorded in consolidated archive','run_date':'not recorded in consolidated archive'})
    df=pd.DataFrame(rows);df.to_csv(OUT/'submission_coverage.csv',index=False)
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Model alias & AD $-$ & AD $+$ & AP $-$ & AP $+$ & Pixels $-$ & Pixels $+$ \\',r'\midrule']
    for model,g in df.groupby('model'):
        cells=[]
        for rep in ['AD','AP','DVRL']:
            for cue in ['no_concept','concept']:
                cell=g[g.representation.eq(rep)&g.cue.eq(cue)]
                if cell.empty: cells.append('--');continue
                v=cell.iloc[0];cells.append(f'{v.valid}'+('' if v.retained else r'$^{a}$'))
        lines.append(tex(display(model))+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (GEN/'coverage.tex').write_text('\n'.join(lines))
    return df

def sensitivity(raw):
    # Repeat representation comparisons under transparent alternative inclusion rules.
    rows=[]
    for min_items,min_fam in [(25,1),(50,2),(100,2),(50,4)]:
        chosen=[]
        for _,g in raw.groupby(['model','representation','concept_flag']):
            v=g.prediction_valid.astype(str).str.lower().eq('true')
            if len(g)>=min_items and g.test_id.nunique()>=min_items and g.dataset_family.nunique()>=min_fam and g.invalid_response_flag.mean()<=.2:
                chosen.append(g)
        if not chosen:
            continue
        f=pd.concat(chosen)
        for invalid_as_error in [False,True]:
            valid=f.prediction_valid.astype(str).str.lower().eq('true')
            ff=f.copy() if invalid_as_error else f[valid].copy()
            scoreable=ff.prediction_valid.astype(str).str.lower().eq('true')
            if invalid_as_error:
                scoreable &= ~ff.invalid_response_flag
            ff['correct']=np.where(scoreable,pd.to_numeric(ff.correct_int).fillna(0),0)
            pairs=ps.machine_representation_pairs(ff)
            for row in pairs.to_dict('records'): rows.append({'min_items':min_items,'min_families':min_fam,'invalid_as_error':invalid_as_error,**row})
    pd.DataFrame(rows).to_csv(OUT/'representation_sensitivity.csv',index=False)
    return rows

def make_model_profiles(m):
    """Render Figure 3 from machine predictions without participant records."""
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    pixel=m[m.representation.eq('DVRL')&m.concept_flag.eq('no_concept')]
    GREY='#C9C9C9'
    # ---- Figure 3. Similar accuracy, opposite criterion: put the accuracy on
    # the plot for the two aliases the text names, grey the rest.
    named3={'gemma4:31b':'#ff7f0e','qwen3-vl:30b':'#2ca02c'}
    fig,axes=plt.subplots(1,2,figsize=(9.5,4.0))
    for model,g in sorted(pixel.groupby('model'),key=lambda kv:kv[0] in named3):
        rates=ps.signal_detection(g,'truth_positive','response_positive')
        pos=g[g.truth_positive].correct.mean()*100;neg=g[~g.truth_positive].correct.mean()*100;acc=g.correct.mean()*100
        c=named3.get(model,GREY);z=3 if model in named3 else 2
        axes[0].plot([0,1],[pos,neg],'o-',color=c,lw=2.0 if c!=GREY else 1.0,zorder=z)
        axes[1].scatter(rates['criterion'],rates['dprime'],s=60 if c!=GREY else 40,color=c,zorder=z)
        if model in named3:
            axes[0].annotate(display(model),(1,neg),xytext=(6,0),textcoords='offset points',va='center',fontsize=8.5,color=c)
            label_position = (0.27, 0.55) if model == 'gemma4:31b' else (0.97, 0.26)
            axes[1].annotate(f'{display(model)}\n{acc:.1f}% correct, $c$={rates["criterion"]:+.2f}',(rates['criterion'],rates['dprime']),
                             xytext=label_position,textcoords='axes fraction',
                             ha='center' if model == 'gemma4:31b' else 'right',va='bottom',fontsize=10,color=c,
                             arrowprops=dict(arrowstyle='-',color=c,lw=.7))
    axes[0].plot([],[],'o-',color=GREY,lw=1,label=f'other {pixel.model.nunique()-len(named3)} aliases');axes[0].legend(frameon=False,fontsize=8,loc='lower left')
    axes[0].set(xticks=[0,1],xticklabels=['Positive queries','Negative queries'],ylabel='Raw-pixel accuracy (%)',ylim=(0,103),xlim=(-.12,1.45))
    axes[0].set_title('A  Error profiles by model',loc='left',fontsize=10)
    axes[1].axvline(0,color='.65',ls='--');axes[1].axhline(0,color='.65',lw=.7)
    axes[1].set(xlabel='Response criterion $c$',ylabel="Sensitivity $d'$",xlim=(-1.2,1.0),ylim=(-.25,1.35))
    axes[1].set_xticks([-1,-0.5,0,0.5,1])
    axes[1].tick_params(axis='y', pad=5)
    axes[1].set_title('B  Sensitivity and response criterion',loc='left',fontsize=10)
    for ax in axes: ax.set_axisbelow(True)
    axes[0].grid(axis='y',alpha=.25)
    fig.tight_layout();savefig(fig,'patterns_model_profiles')


def make_figures(h,m,sdt,results):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    GREY='#C9C9C9'
    # ---- Figure 2. Thirteen colours were not separable, so every alias is grey
    # and only the three the text discusses carry colour and a direct label.
    named2={'gemini-3.5-flash':'#1f77b4','gpt-5.5':'#d62728','qwen3-vl:30b':'#2ca02c'}
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.7),gridspec_kw={'width_ratios':[0.8,1.6]})
    for y,key in [(1,'dprime'),(0,'criterion')]:
        v=results['human_signal_detection'][key];d=v['difference'];lo,hi=v['ci95']
        axes[0].errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt='o',color='#287B8E',capsize=4)
    axes[0].axvline(0,color='.6',ls='--');axes[0].set(yticks=[0,1],yticklabels=['Criterion $c$',"Sensitivity $d'$"],ylim=(-.6,1.6),xlabel='Cue minus no cue')
    axes[0].set_title('A  Human cue contrasts',loc='left',fontsize=10)
    order=sorted(sdt.itertuples(),key=lambda r:r.model in named2)
    for r in order:
        c=named2.get(r.model,GREY);lw=2.0 if r.model in named2 else 1.0
        axes[1].plot([0,1],[r.pixel_dprime,r.description_dprime],'o-',color=c,lw=lw,ms=4 if c==GREY else 5,zorder=3 if c!=GREY else 2)
        if r.model in named2:
            axes[1].annotate(display(r.model),(1,r.description_dprime),xytext=(8,0),textcoords='offset points',va='center',fontsize=8.5,color=c)
    axes[1].plot([],[],'o-',color=GREY,lw=1,ms=4,label=f'other {len(sdt)-len(named2)} aliases');axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    axes[1].set(xticks=[0,1],xticklabels=['Raw pixels','Action Descriptions'],ylabel="Sensitivity $d'$",xlim=(-.15,1.55))
    axes[1].set_title(f'B  {len(sdt)} models on matched items',loc='left',fontsize=10)
    for ax in axes: ax.set_axisbelow(True)
    axes[1].grid(axis='y',alpha=.25)
    fig.tight_layout();savefig(fig,'patterns_signal_detection')

    make_model_profiles(m)

    make_task_design_figure()


def make_task_design_figure():
    # ---- Figure 1. Design overview: one item, the human arms with a real cue,
    # and the machine representation ladder, cue levels, runs and outcomes.
    fig,ax=plt.subplots(figsize=(10,5.9));ax.set(xlim=(0,20),ylim=(0,11.4));ax.axis('off')
    def box(x,y,w,hh,text,fc,fs=11.5):
        ax.add_patch(FancyBboxPatch((x,y),w,hh,boxstyle='round,pad=0.06',facecolor=fc,edgecolor='.55',linewidth=.8))
        ax.text(x+w/2,y+hh/2,text,ha='center',va='center',fontsize=fs)
    def arrow(x0,y0,x1,y1): ax.annotate('',(x1,y1),(x0,y0),arrowprops=dict(arrowstyle='->',color='.4',lw=1))
    def head(x,y,t): ax.text(x,y,t,fontsize=12,fontweight='bold',va='bottom')
    # left column: 0.1 .. 8.4
    head(0.1,10.75,'Task: 200 queries, 4 families')
    box(0.1,9.3,3.0,1.1,'6 positive\nexamples','#E8F1F3');box(3.3,9.3,3.0,1.1,'6 negative\nexamples','#F5EDE5');box(6.5,9.3,1.9,1.1,'1 query','#F5F5F5')
    head(0.1,7.8,'Humans: 207 students, 20 items each')
    box(0.1,6.3,4.0,1.1,'No cue\n(98 participants)','#F5F5F5');box(4.4,6.3,4.0,1.1,'Concept cue\n(109 participants)','#E8F1F3')
    ax.text(4.4,5.98,'example: "has three\nstraight lines,\nhas two parts"',fontsize=10.5,style='italic',va='top')
    box(0.1,3.8,8.3,.9,'Positive / negative query decision','#FFFFFF')
    head(0.1,1.6,'Outcomes')
    box(0.1,0.3,4.0,1.0,"sensitivity $d'$\n(discrimination)",'#FFFFFF');box(4.4,0.3,4.0,1.0,'criterion $c$\n(endorsement)','#FFFFFF')
    ax.plot([9.0,9.0],[0.2,11.1],color='.8',lw=1)
    # right column: 9.5 .. 19.9
    head(9.5,10.75,'Models: 15 compared, 4 factorial')
    ax.text(9.5,10.25,'Factorial: 2 proprietary + 2 open-weight',fontsize=10.5)
    ax.text(9.5,9.65,'Representations of the same 13 figures',fontsize=11)
    rungs=[('Raw pixels','#F5F5F5'),('Own step\ncaptions','#FFF3D6'),('Own prose\ncaptions*','#FFF3D6'),
           ('Benchmark\ndescriptions','#E8F1F3'),('Renderer-faithful\ntext*','#E8F1F3'),('Benchmark\nprograms','#E8F1F3')]
    for i,(t,fc) in enumerate(rungs):
        box(9.5+(i%3)*3.45,8.0-(i//3)*1.4,3.2,1.1,t,fc,fs=10.5)
    ax.text(9.5,6.25,'GPT also reasons from Gemini captions',fontsize=9.5,va='top')
    ax.text(9.5,5.65,'Concept cue',fontsize=11)
    for j,(t,fc) in enumerate([('none','#F5F5F5'),('correct label','#E8F1F3'),('original\nmismatched','#F5EDE5')]):
        box(9.5+j*3.45,4.25,3.2,1.0,t,fc,fs=10.5)
    box(9.5,3.3,4.9,.65,'clean mismatched*','#F5EDE5',fs=10)
    box(14.7,3.3,5.1,.65,'irrelevant text*','#F5F5F5',fs=10)
    ax.text(9.5,3.05,'*Added in open-weight factorial; not every combination',fontsize=9,va='top')
    ax.text(9.5,2.65,'15-model study: 1 prediction per query / interface\nGemini: 1 greedy + 3 sampled runs\nGPT: 2 sampled runs\nGemma / Qwen3.6: 1 greedy + 1 sampled run',fontsize=9,va='top',linespacing=1.1)
    box(9.5,0.3,4.9,1.0,"$d'$ and $c$ per model,\nrepresentation and cue",'#FFFFFF',fs=10.5)
    box(14.7,0.3,5.1,1.0,'coverage and failures\nper cell','#FFFFFF',fs=10.5)
    fig.tight_layout();savefig(fig,'patterns_task_design')

def write_tables(h,sdt,results,alignment):
    cells=h.groupby(['category','truth_positive','cue']).agg(trials=('correct','size'),accuracy=('correct','mean')).reset_index()
    cells.to_csv(OUT/'human_condition_family_polarity.csv',index=False)
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Family & Query & No-cue $n$ & No-cue (\%) & Cue $n$ & Cue (\%) \\',r'\midrule']
    for (fam,pol),g in cells.groupby(['category','truth_positive']):
        a=g[g.cue.eq(0)].iloc[0];b=g[g.cue.eq(1)].iloc[0]
        lines.append(f"{FAMILIES[fam]} & {'Positive' if pol else 'Negative'} & {a.trials} & {a.accuracy*100:.1f} & {b.trials} & {b.accuracy*100:.1f}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(GEN/'human_cells.tex').write_text('\n'.join(lines))
    lines=[r'\begin{tabular}{rrrrrrr}',r'\toprule',r'Cap (s) & Trials & Positive change (\%) & 95\% CI & Negative change (\%) & 95\% CI & Interaction $p$ \\',r'\midrule']
    for cap in [120,180,300]:
        v=results['human_rt'][f'cap_{cap}_seconds'];a=v['terms']['cue'];b=v['terms']['net_cue_effect_negative']
        lines.append(f"{cap} & {v['n_trials']} & {a['percent_change']:+.1f} & [{a['ci95_percent_change'][0]:.1f}, {a['ci95_percent_change'][1]:.1f}] & {b['percent_change']:+.1f} & [{b['ci95_percent_change'][0]:.1f}, {b['ci95_percent_change'][1]:.1f}] & {ptex(v['terms']['cue:negative']['p'])}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(GEN/'rt_cutoffs.tex').write_text('\n'.join(lines))
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Model & Items & Pixel $d\prime$ & Description $d\prime$ & $\Delta d\prime$ [95\% CI] \\',r'\midrule']
    for _,r in sdt.iterrows():
        lines.append(f"{tex(display(r.model))} & {r.n_items} & {r.pixel_dprime:.2f} & {r.description_dprime:.2f} & {r.delta_dprime:+.2f} [{r.delta_dprime_low:.2f}, {r.delta_dprime_high:.2f}]"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(GEN/'paired_sdt.tex').write_text('\n'.join(lines))
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Comparison & Items & Overall $r_s$ & Positive $r_s$ & Negative $r_s$ \\',r'\midrule']
    for r in alignment['sensitivity']:
        lines.append(f"{r['comparison']} & {r['n_items']} & {r['overall']:.3f} & {r['positive']:.3f} & {r['negative']:.3f}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(GEN/'alignment_sensitivity.tex').write_text('\n'.join(lines))

def main():
    GEN.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    h=ps.prepare_human();m=ps.prepare_machine();raw=pd.read_csv(ps.MACHINE,low_memory=False)
    assert not m.duplicated(['model','representation','concept_flag','test_id']).any()
    assert len(h)==4140 and h.participant_id.nunique()==207
    results=json.loads((OUT/'patterns_statistics.json').read_text())
    assert results['analysis_version']=='patterns_v2_20260907'
    sdt=paired_sdt(m); print('Paired SDT complete',flush=True)
    a=alignment_table(h,m);print('Alignment complete',flush=True)
    coverage(raw);sens=sensitivity(raw);write_tables(h,sdt,results,a);make_figures(h,m,sdt,results)
    summary={'seed':SEED,'bootstrap_iterations':NBOOT,'bootstrap_unit':'base problem within family; recorded responses fixed',
             'human_unique_queries':h.question_id.nunique(),'human_base_problems':decorate(h,'question_id').base.nunique(),
             'paired_sdt':sdt.to_dict('records'),'alignment':a,'representation_sensitivity':sens}
    (OUT/'submission_checks.json').write_text(json.dumps(ps.native(summary),indent=2))
    import scipy,statsmodels
    manifest={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,
              'scipy':scipy.__version__,'statsmodels':statsmodels.__version__,'matplotlib':matplotlib.__version__,
              'sha256':{str(p.relative_to(ps.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ps.HUMAN,ps.MACHINE,Path(ps.__file__),Path(__file__)]}}
    (OUT/'submission_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(ps.native(summary['alignment']),indent=2))

if __name__=='__main__':
    if sys.argv[1:] == ['--task-design-only']:
        make_task_design_figure()
    elif sys.argv[1:] == ['--model-profiles-only']:
        make_model_profiles(ps.prepare_machine())
    else:
        main()
