"""Make publication-exportable SVG and PNG figures from measured JSONL data."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT=Path(__file__).parent/'parallel_results'
COLORS={'exhaustive':'#2563a6','inundation':'#d05b27'}
LABELS={'exhaustive':'Полный перебор + полные потоки',
        'inundation':'Дерево + точный PDownsize с порогом'}


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT)
    parser.add_argument('--metric',choices=['seconds','cpu_seconds'],default='seconds')
    args=parser.parse_args()
    ROOT=args.input
    rows=[json.loads(s) for s in (ROOT/'results.jsonl').read_text().splitlines()]
    metadata=json.loads((ROOT/'metadata.json').read_text())
    metric=args.metric
    time_label='Время выполнения' if metric=='seconds' else 'Процессорное время'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none'})
    large = any(row['case'][1] == 48 for row in rows)
    varying_r_n = 28 if large else (24 if any(row['case'][0]=='ER' and row['case'][1]==24 and row['case'][2]==10 for row in rows) else 20)
    varying_n_r = 6 if large else 8
    sbm_n = 28 if large else 20
    panels=[
        (f'Зависимость от n: ER, r = {varying_n_r}, p = 0.5','Число вершин n',1,
         lambda c:c[0]=='ER' and c[2]==varying_n_r and c[3]==0.5),
        (f'Зависимость от r: ER, n = {varying_r_n}, p = 0.5','Размер подграфа r',2,
         lambda c:c[0]=='ER' and c[1]==varying_r_n and c[3]==0.5),
        (f'Два сообщества: n = {sbm_n}, pᵢₙ = 0.65, pₒᵤₜ = 0.15','Размер подграфа r',2,
         lambda c:c[0]=='SBM' and c[1]==sbm_n),
    ]
    fig,axes=plt.subplots(1,3,figsize=(17,5.5),layout='constrained')
    for ax,(title,xlabel,index,predicate) in zip(axes.flat,panels):
        selected=[row for row in rows if predicate(row['case'])]
        xs=sorted({row['case'][index] for row in selected})
        for method in COLORS:
            groups=[[row[method][metric] for row in selected if row['case'][index]==x] for x in xs]
            mid=[median(g) for g in groups]
            ax.plot(xs,mid,'o-',color=COLORS[method],label=LABELS[method],lw=2)
            ax.fill_between(xs,[min(g) for g in groups],[max(g) for g in groups],
                            color=COLORS[method],alpha=0.13)
        ax.set(title=title,xlabel=xlabel,ylabel=f'{time_label}, с (лог. шкала)',yscale='log')
        ax.set_xticks(xs)
        ax.grid(True,which='both',alpha=0.18)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('Два точных метода максимизации вершинной связности\n'
                 f'Медиана трёх графов; полоса — min–max. Дерево включено. Рабочих процессов: {metadata.get("workers",1)}.',fontsize=14)
    for ext in ('png','svg'):
        fig.savefig(ROOT/f'{"timings" if metric=="seconds" else "timings_cpu"}.{ext}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    chosen=[row for row in rows if row['case'][0]=='ER' and row['case'][1]==varying_r_n and row['case'][3]==0.5]
    rs=sorted({row['case'][2] for row in chosen})
    for method in COLORS:
        values=[median([row[method]['stats']['flows'] for row in chosen if row['case'][2]==r]) for r in rs]
        axes[0].plot(rs,values,'o-',color=COLORS[method],label=LABELS[method])
    axes[0].set(title='Число запусков потока, включая дерево',xlabel='r',ylabel='Потоки, логарифмическая шкала',yscale='log')
    builds=[median([row['inundation']['build_seconds'] for row in chosen if row['case'][2]==r]) for r in rs]
    searches=[median([row['inundation']['search_seconds'] for row in chosen if row['case'][2]==r]) for r in rs]
    axes[1].plot(rs,builds,'o-',label='Построение дерева',color='#7254a3')
    axes[1].plot(rs,searches,'o-',label='Поиск PDownsize / обход',color='#d05b27')
    axes[1].set(title='Из чего состоит время второго метода',xlabel='r',ylabel='Секунды, логарифмическая шкала',yscale='log')
    for ax in axes:
        ax.grid(True,which='both',alpha=0.18)
        ax.set_xticks(rs)
        ax.legend(fontsize=8)
    fig.suptitle(f'ER, n = {varying_r_n}, p = 0.5; медианы трёх реализаций')
    for ext in ('png','svg'):
        fig.savefig(ROOT/f'work.{ext}',dpi=180)
    plt.close(fig)
    flat=[]
    for row in rows:
        family,n,r,p,seed=row['case']
        item=dict(family=family,n=n,r=r,p=p,seed=seed,edges=row['edges'],subsets=row['subsets'],k=row['exhaustive']['k'])
        for method in COLORS:
            item[method+'_seconds']=row[method]['seconds']
            item[method+'_cpu_seconds']=row[method].get('cpu_seconds','')
            for key,value in row[method]['stats'].items():
                item[method+'_'+key]=value
        item['tree_seconds']=row['inundation']['build_seconds']
        item['search_seconds']=row['inundation']['search_seconds']
        item['speedup']=row['exhaustive']['seconds']/row['inundation']['seconds']
        flat.append(item)
    with (ROOT/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    groups={}
    for row in flat:
        groups.setdefault((row['family'],row['n'],row['r'],row['p']),[]).append(row)
    lines=['Время выполнения (wall clock), медианы трёх графов. Построение дерева включено.', '',
           '| Семейство | n | r | p | Полный перебор, с | Дерево + PDownsize, с | Ускорение¹ | k по seeds |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for case,group in sorted(groups.items()):
        family,n,r,p=case
        p_label='0.65 / 0.15' if family=='SBM' else str(p)
        a=median(x['exhaustive_seconds'] for x in group)
        b=median(x['inundation_seconds'] for x in group)
        ratio=median(x['speedup'] for x in group)
        values=', '.join(str(x['k']) for x in sorted(group,key=lambda x:x['seed']))
        lines.append(f'| {family} | {n} | {r} | {p_label} | {a:.3f} | {b:.3f} | {ratio:.1f}× | {values} |')
    lines.extend(['','¹ Медиана отношений времён на одном и том же графе, а не отношение медиан.',
                  'Для SBM столбец p показывает вероятности внутри сообщества / между сообществами.',
                  f'Все {len(rows)} пар ответов совпали; каждый свидетель дополнительно проверен полными потоками.',
                  f'Сумма wall time отдельных методов: {sum(x[m]["seconds"] for x in rows for m in COLORS):.1f} с (при параллельном запуске это НЕ длительность серии).'])
    sessions=ROOT/'sessions.jsonl'
    if sessions.exists():
        lines.append(f'Время завершённых сессий запуска: {sum(json.loads(s)["wall_seconds"] for s in sessions.read_text().splitlines()):.1f} с.')
    if all('cpu_seconds' in row['exhaustive'] for row in rows):
        lines.extend(['','Процессорное время (process_time), медианы:', '',
                      '| Семейство | n | r | p | Полный перебор, CPU с | Дерево + PDownsize, CPU с |',
                      '|---|---:|---:|---:|---:|---:|'])
        for case,group in sorted(groups.items()):
            family,n,r,p=case
            p_label='0.65 / 0.15' if family=='SBM' else str(p)
            a=median(x['exhaustive_cpu_seconds'] for x in group)
            b=median(x['inundation_cpu_seconds'] for x in group)
            lines.append(f'| {family} | {n} | {r} | {p_label} | {a:.3f} | {b:.3f} |')
    (ROOT/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
