# ogbg-molhiv

To train baselines with FLAG in the default setup, run

**GCN+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).
        
    python main_pyg.py --dataset ogbg-molhiv --gnn gcn --step-size 1e-2

**GCN+V+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).

    python main_pyg.py --dataset ogbg-molhiv --gnn gcn-virtual --step-size 1e-3

**GIN+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).

    python main_pyg.py --dataset ogbg-molhiv --gnn gin --step-size 5e-3

**GIN+V+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).

    python main_pyg.py --dataset ogbg-molhiv --gnn gin-virtual --step-size 1e-3


# ogbg-molpcba

To train baselines with FLAG in the default setup, run

**GCN+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).
        
    python main_pyg.py --dataset ogbg-molpcba --gnn gcn --step-size 8e-3

**GCN+V+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).

    python main_pyg.py --dataset ogbg-molpcba --gnn gcn-virtual --step-size 8e-3

**GIN+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).

    python main_pyg.py --dataset ogbg-molpcba --gnn gin --step-size 8e-3

**GIN+V+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).

    python main_pyg.py --dataset ogbg-molpcba --gnn gin-virtual --step-size 8e-3

