# Acknowledgements

## Upstream projects

Battery-TA-RWARE grew out of two open-source environments, both released under the MIT licence:

- **TA-RWARE** (task-assignment robotic warehouse), by Raul Steleac and co-authors:
  <https://github.com/raulsteleac/task-assignment-robotic-warehouse>.
  The licence notice `Copyright (c) 2024 Raul Steleac` is kept in [LICENSE](LICENSE).
  > A. Krnjaic, R. D. Steleac, J. D. Thomas, G. Papoudakis, L. Schäfer, A. W. K. To, K.-H. Lao,
  > M. Cubuktepe, M. Haley, P. Börsting, S. V. Albrecht. *Scalable Multi-Agent Reinforcement
  > Learning for Warehouse Logistics with Robotic and Human Co-Workers.* arXiv:2212.11498, 2022.
- **RWARE** (multi-robot warehouse), by Filippos Christianos and co-authors:
  <https://github.com/semitable/robotic-warehouse>.
  > G. Papoudakis, F. Christianos, L. Schäfer, S. V. Albrecht. *Benchmarking Multi-Agent Deep
  > Reinforcement Learning Algorithms in Cooperative Tasks.* NeurIPS Datasets and Benchmarks, 2021.

Battery-TA-RWARE v1 (branch `v1`, tag `legacy-v1.1.0`) extended TA-RWARE with battery management.
Version 2 is a new engine: its simulator, task model and movement code were written from scratch
and share no code with TA-RWARE or RWARE. It keeps the warehouse setting and the idea of
heterogeneous robots that have to cooperate.

## Algorithms

The macro-mode movement uses PIBT. The implementation was written for this package; no code was
copied from the reference implementations. The algorithms come from:

- K. Okumura, M. Machida, X. Défago, Y. Tamura. *Priority Inheritance with Backtracking for
  Iterative Multi-Agent Path Finding.* Artificial Intelligence 310, 103752, 2022.
  doi:10.1016/j.artint.2022.103752
- K. Okumura. *Improving LaCAM for Scalable Eventually Optimal Multi-Agent Pathfinding.* IJCAI 2023.
  The *swap* operation; here it is restricted to robots trapped in dead ends.

The dispatching rules offered by the greedy baseline (FIFO, EDD, SPT, critical ratio and priority)
are standard scheduling heuristics.

## Dependencies

[NumPy](https://numpy.org), [Gymnasium](https://gymnasium.farama.org) and, optionally,
[PettingZoo](https://pettingzoo.farama.org), [PyYAML](https://pyyaml.org),
[Matplotlib](https://matplotlib.org) and [Pillow](https://python-pillow.org).
