"""Draw the saved benchmark inputs, without regenerating any graph."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).parent / 'parallel_results'
OUT = ROOT / 'graph_visualizations'
SEEDS = (101, 202, 303)


def draw(ax, family, n, p, seed):
    data = json.loads((ROOT / 'graphs' / f'{family}_n{n}_p{p}_seed{seed}.json').read_text())
    assert data['n'] == n
    edges = data['edges']
    if family == 'ER':
        positions = [(math.cos(math.pi/2+2*math.pi*v/n),
                      math.sin(math.pi/2+2*math.pi*v/n)) for v in range(n)]
        colors = ['#2678a5'] * n
        ax.add_collection(LineCollection([(positions[u], positions[v]) for u,v in edges],
                                         colors='#8195a5', linewidths=0.65, alpha=0.48))
    else:
        positions = [((-0.92 if v < n//2 else 0.92) + 0.64*math.cos(math.pi/2+2*math.pi*(v%(n//2))/(n//2)),
                      0.64*math.sin(math.pi/2+2*math.pi*(v%(n//2))/(n//2))) for v in range(n)]
        colors = ['#2678a5' if v < n//2 else '#17846b' for v in range(n)]
        for inside, color, alpha in [(False,'#d58c31',0.40),(True,'#677c8a',0.65)]:
            segments = [(positions[u],positions[v]) for u,v in edges
                        if ((u < n//2) == (v < n//2)) == inside]
            ax.add_collection(LineCollection(segments, colors=color, linewidths=0.8, alpha=alpha))
    x,y = zip(*positions)
    ax.scatter(x,y,s=170 if family=='ER' else 210,c=colors,edgecolors='white',linewidths=0.9,zorder=3)
    for v,(x,y) in enumerate(positions):
        ax.text(x,y,str(v),ha='center',va='center',fontsize=6.7,color='white',zorder=4)
    m=len(edges)
    density=2*m/(n*(n-1))
    ax.set_title(f'n = {n}, m = {m}, плотность = {density:.2f}\nseed = {seed}',fontsize=11,pad=9)
    ax.set_aspect('equal')
    ax.set_xlim((-1.2,1.2) if family=='ER' else (-1.75,1.75))
    ax.set_ylim((-1.2,1.2) if family=='ER' else (-0.92,0.92))
    ax.axis('off')
    return {'family':family,'n':n,'p':p,'seed':seed,'m':m,'density':density}


def main():
    global ROOT, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT)
    ROOT=parser.parse_args().input
    OUT=ROOT/'graph_visualizations'
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    metadata=json.loads((ROOT/'metadata.json').read_text())
    large=any(c[1]==48 for c in metadata['configurations'])
    sizes=(24,28,32,40,48) if large else (18,20,22)
    fixed_n=28 if large else 20
    manifest=[]
    for name, values, title in [
        ('er_sizes',sizes,'Эрдёш — Реньи: изменение размера n, вероятность ребра p = 0.5'),
        ('er_density',(0.25,0.5,0.75),f'Эрдёш — Реньи: изменение вероятности ребра p, n = {fixed_n}'),
    ]:
        fig,axes=plt.subplots(len(values),3,figsize=(12,4*len(values)),layout='constrained')
        for row,value in enumerate(values):
            for col,seed in enumerate(SEEDS):
                n,p=(value,0.5) if name=='er_sizes' else (fixed_n,value)
                manifest.append(draw(axes[row,col],'ER',n,p,seed))
                if col==0:
                    axes[row,col].text(-0.10,0.5,f'n = {n}' if name=='er_sizes' else f'p = {p}',
                                       transform=axes[row,col].transAxes,rotation=90,
                                       va='center',fontsize=13,fontweight='bold')
        fig.suptitle(title+'\nТочные входы эксперимента; фиксированная круговая раскладка',fontsize=15)
        for ext in ('png','svg'):
            fig.savefig(OUT/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,seed in zip(axes,SEEDS):
        manifest.append(draw(ax,'SBM',fixed_n,0.4,seed))
    fig.suptitle(f'Два сообщества по {fixed_n//2} вершин: p внутри = 0.65, p между = 0.15\n'
                 'Цвет вершин — заданная группа; оранжевые рёбра соединяют группы',fontsize=14)
    for ext in ('png','svg'):
        fig.savefig(OUT/f'sbm.{ext}',dpi=180)
    plt.close(fig)
    unique={(x['family'],x['n'],x['p'],x['seed']) for x in manifest}
    expected={(c[0],c[1],c[3],c[4]) for c in metadata['configurations']}
    assert unique==expected
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if large:
        fig,axes=plt.subplots(2,3,figsize=(12,8.5),layout='constrained')
        for ax,n in zip(axes.flat,sizes):
            draw(ax,'ER',n,0.5,101)
        draw(axes.flat[-1],'SBM',fixed_n,0.4,101)
        fig.suptitle('Новая серия: 24–48 вершин\nПримеры точных входных графов, seed = 101; снизу справа — два сообщества',fontsize=14)
        for ext in ('png','svg'):
            fig.savefig(OUT/f'overview.{ext}',dpi=180)
        plt.close(fig)
    print(f'Rendered all {len(unique)} saved input graphs; n={fixed_n},p=0.5 appears in both ER figures.')


if __name__=='__main__':
    main()
