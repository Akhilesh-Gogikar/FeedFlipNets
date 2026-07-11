# Dependency Graph

## Most Imported Files (change these carefully)

- `/utils.py` — imported by **15** files
- `//core/types.py` — imported by **15** files
- `/registry.py` — imported by **11** files
- `/data/loaders.py` — imported by **5** files
- `/core.py` — imported by **4** files
- `/types.py` — imported by **4** files
- `/loaders.py` — imported by **4** files
- `/cache.py` — imported by **3** files
- `///core/types.py` — imported by **3** files
- `//registry.py` — imported by **3** files
- `//utils.py` — imported by **3** files
- `/transformer.py` — imported by **2** files
- `//core/deep_mlp.py` — imported by **2** files
- `//core/transformer.py` — imported by **2** files
- `/metrics.py` — imported by **2** files
- `/data.py` — imported by **2** files
- `/trainer.py` — imported by **2** files
- `//core/strategies.py` — imported by **2** files
- `/mnist.py` — imported by **1** files
- `/timeseries.py` — imported by **1** files

## Import Map (who imports what)

- `/utils.py` ← `datasets/__init__.py`, `datasets/mnist.py`, `datasets/timeseries.py`, `datasets/tinystories.py`, `feedflipnets/__init__.py` +10 more
- `//core/types.py` ← `feedflipnets/data/adult.py`, `feedflipnets/data/ag_news.py`, `feedflipnets/data/california.py`, `feedflipnets/data/cifar10.py`, `feedflipnets/data/csv_generic.py` +10 more
- `/registry.py` ← `feedflipnets/data/__init__.py`, `feedflipnets/data/adult.py`, `feedflipnets/data/ag_news.py`, `feedflipnets/data/california.py`, `feedflipnets/data/cifar10.py` +6 more
- `/data/loaders.py` ← `feedflipnets/utils.py`, `feedflipnets/utils.py`, `feedflipnets/utils.py`, `feedflipnets/utils.py`, `feedflipnets/utils.py`
- `/core.py` ← `feedflipnets/__init__.py`, `feedflipnets/__init__.py`, `feedflipnets/__init__.py`, `feedflipnets/__init__.py`
- `/types.py` ← `feedflipnets/core/activations.py`, `feedflipnets/core/deep_mlp.py`, `feedflipnets/core/quant.py`, `feedflipnets/core/strategies.py`
- `/loaders.py` ← `feedflipnets/data/__init__.py`, `feedflipnets/data/__init__.py`, `feedflipnets/data/__init__.py`, `feedflipnets/data/__init__.py`
- `/cache.py` ← `feedflipnets/data/ag_news.py`, `feedflipnets/data/mnist.py`, `feedflipnets/data/ucr.py`
- `///core/types.py` ← `feedflipnets/data/loaders/synth_fixture.py`, `feedflipnets/data/loaders/synthetic.py`, `feedflipnets/data/loaders/tinystories.py`
- `//registry.py` ← `feedflipnets/data/loaders/synth_fixture.py`, `feedflipnets/data/loaders/synthetic.py`, `feedflipnets/data/loaders/tinystories.py`
