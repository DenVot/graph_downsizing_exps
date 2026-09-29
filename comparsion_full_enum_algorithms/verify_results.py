"""Independently verify recorded witnesses by vertex deletion, and counters."""
import argparse
import json
import math
from pathlib import Path
from algorithms import graph_from_edges, induced
from test_algorithms import oracle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).parent/'parallel_results')
    root=parser.parse_args().input
    rows=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
    cases=json.loads((root/'metadata.json').read_text())['configurations']
    assert len(rows)==len(cases)
    assert {tuple(row['case']) for row in rows}=={tuple(c) for c in cases}
    for row in rows:
        data=json.loads((root/'graphs'/row['graph']).read_text())
        graph=graph_from_edges(data['n'],data['edges'])
        r=row['case'][2]
        assert row['exhaustive']['stats']['subsets']==math.comb(len(graph),r)
        assert row['exhaustive']['stats']['flows']==math.comb(len(graph),r)*math.comb(r,2)
        for method in ('exhaustive','inundation'):
            result=row[method]
            assert len(set(result['witness']))==r
            assert oracle(induced(graph,result['witness']))==result['k']
            assert result['cpu_seconds']>0 and result['seconds']>0
    summary=dict(cases=len(rows),witnesses=2*len(rows),
                 independent_oracle='enumeration of vertex cuts + graph traversal',
                 baseline_subset_and_flow_counts='all matched',status='passed')
    (root/'verification.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
