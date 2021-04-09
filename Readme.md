## Introduction
D3PE (Deep Data-Driven Policy Evaluation) aims to evaluation a large set of candidate policies by a fix dataset to select best ones.

## Installation
```bash
bash install.sh
```

## Benchmark
First you need to download the models. On a Suzhou machine:

```bash
scp 10.200.0.10:~/data/ope_benchmarks.tar ./
tar -xvf ope_benchmarks.tar
```

Then you can lauch the ope algorithm to evaluate the policies of certain task in the benchmarks, e.g.

```bash
python scripts/lauch_ope.py --domain HalfCheetah-v3 --level low --amount 99 -on online -oa online
```

Then you can evaluate the result of that algorithm by:

```bash
python scripts/eval_ope.py -d HalfCheetah-v3 --level low --amount 99 -en online
```