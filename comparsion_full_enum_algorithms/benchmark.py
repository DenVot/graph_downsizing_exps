"""Parallel cases, sequential paired methods; resumable JSONL and saved inputs."""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from itertools import combinations
from pathlib import Path
from time import time, process_time, perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from algorithms import graph_from_edges, components, exhaustive, inundation, induced, connectivity


def synthetic(n, p, seed, family):
    rng = random.Random(seed)
    for attempt in range(10000):
        edges = []
        for u, v in combinations(range(n), 2):
            probability = p if family == 'ER' else (0.65 if (u < n//2) == (v < n//2) else 0.15)
            if rng.random() < probability:
                edges.append((u, v))
        graph = graph_from_edges(n, edges)
        if len(components(graph, range(n))) == 1:
            return graph, edges, attempt
    raise RuntimeError('Could not generate connected graph')


def configurations(extended=False, large=False):
    cases = set()
    if large:
        for seed in (101, 202, 303):
            for n in (24, 28, 32, 40, 48):
                cases.add(('ER', n, 6, 0.5, seed))
            for r in (4, 5, 6, 7, 8):
                cases.add(('ER', 28, r, 0.5, seed))
            for p in (0.25, 0.5, 0.75):
                cases.add(('ER', 28, 6, p, seed))
            for r in (6, 8):
                cases.add(('SBM', 28, r, 0.4, seed))
        return sorted(cases, key=lambda x:(math.comb(x[1],x[2])*x[2]**2, x))
    for seed in (101, 202, 303):
        for n in ((18,20,22,24,26) if extended else (18,20,22)):
            cases.add(('ER',n,8,0.5,seed))
        for r in ((4,6,8,10,12) if not extended else (4,6,8,10,12,14,16)):
            cases.add(('ER',20 if not extended else 24,r,0.5,seed))
        for p in (0.25,0.5,0.75):
            cases.add(('ER',20,8,p,seed))
        for r in (6,8,10):
            cases.add(('SBM',20,r,0.4,seed))
    return sorted(cases, key=lambda x:(math.comb(x[1],x[2])*x[2]**2, x))


def warmup():
    warm,_,_=synthetic(9,0.5,42,'ER')
    exhaustive(warm,5); inundation(warm,5)


def run_case(index, case):
    family,n,r,p,seed=case
    graph,edges,attempt=synthetic(n,p,seed,family)
    graph_name=f'{family}_n{n}_p{p}_seed{seed}.json'
    print('START',case,'subsets',math.comb(n,r),flush=True)
    results={}
    methods=[('exhaustive',exhaustive),('inundation',inundation)]
    if index%2:
        methods.reverse()
    for name,method in methods:
        cpu_start=process_time()
        results[name]=method(graph,r)
        results[name]['cpu_seconds']=process_time()-cpu_start
        print(' METHOD',case,name,round(results[name]['seconds'],3),'k',results[name]['k'],flush=True)
    assert results['exhaustive']['k']==results['inundation']['k'],(case,results)
    for result in results.values():
        assert len(set(result['witness']))==r
        assert connectivity(induced(graph,result['witness']))==result['k']
    row=dict(case=case,edges=len(edges),subsets=math.comb(n,r),graph=graph_name,
             method_order=[x[0] for x in methods],finished_unix=time(),**results)
    print('DONE',case,'speedup',round(results['exhaustive']['seconds']/results['inundation']['seconds'],2),flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'parallel_results')
    profiles=parser.add_mutually_exclusive_group()
    profiles.add_argument('--extended',action='store_true')
    profiles.add_argument('--large',action='store_true',help='24–48 vertices, 39 paired cases')
    parser.add_argument('--workers',type=int,default=min(8,os.cpu_count() or 1))
    args=parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'graphs').mkdir(exist_ok=True)
    source_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob('*.py')}
    metadata=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
                  processor=platform.processor(),source_sha256=source_hashes,
                  workers=args.workers,logical_cpus=os.cpu_count(),
                  clock='perf_counter wall seconds; process_time CPU seconds; sequential methods per worker',
                  configurations=configurations(args.extended,args.large))
    meta_path=args.output/'metadata.json'
    if meta_path.exists():
        old=json.loads(meta_path.read_text())
        if old['source_sha256'].get('algorithms.py') != source_hashes['algorithms.py']:
            raise RuntimeError('Algorithm changed: use a fresh output directory')
        if old.get('workers') != args.workers:
            raise RuntimeError('Worker count changed: use a fresh output directory')
        if old['configurations'] != [list(c) for c in metadata['configurations']]:
            raise RuntimeError('Configuration grid changed: use a fresh output directory')
    else:
        meta_path.write_text(json.dumps(metadata,indent=2))
    output=args.output/'results.jsonl'
    rows=[json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    done={tuple(row['case']) for row in rows}
    pending=[(i,c) for i,c in enumerate(configurations(args.extended,args.large)) if c not in done]
    for _,case in pending:
        family,n,r,p,seed=case
        _,edges,attempt=synthetic(n,p,seed,family)
        graph_name=f'{family}_n{n}_p{p}_seed{seed}.json'
        (args.output/'graphs'/graph_name).write_text(json.dumps(dict(n=n,edges=edges,attempt=attempt)))
    # Largest estimated jobs first to reduce idle workers at the end.
    pending.sort(key=lambda x:math.comb(x[1][1],x[1][2])*x[1][2]**5,reverse=True)
    started=perf_counter()
    with output.open('a',buffering=1) as stream:
        if args.workers == 1:
            warmup()
            for index,case in pending:
                stream.write(json.dumps(run_case(index,case))+'\n')
        else:
            with ProcessPoolExecutor(max_workers=args.workers,initializer=warmup) as pool:
                futures=[pool.submit(run_case,index,case) for index,case in pending]
                for future in as_completed(futures):
                    stream.write(json.dumps(future.result())+'\n')
    with (args.output/'sessions.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(cases_completed=len(pending),wall_seconds=perf_counter()-started,
                                     workers=args.workers,finished_unix=time()))+'\n')


if __name__=='__main__':
    main()
