# Attribution

- **Scaling Group Inference** — Gaurav Parmar et al., [paper](https://arxiv.org/abs/2508.15773), [code](https://github.com/GaParmar/group-inference). Vendored at `reproduction/third_party/sgi`, MIT license retained.
- **SatisDive** — Kevin Zhai et al., [paper](https://arxiv.org/abs/2610.02372), [code](https://github.com/UCF-CRCV/SatisDive). Vendored at `reproduction/third_party/satisdive`, Apache 2.0 license retained.
- SatisDive's bundled **HPSv3**, including its differentiable preprocessing, retains its own MIT license and upstream attribution. See the vendored `THIRD_PARTY.md` and `third_party/HPSv3/LICENSE`.
- The two versioned paper PDFs in `papers/` are the authors' arXiv v1 publications, listed as CC BY 4.0 on their arXiv HTML pages. They are research references, not benchmark outputs.
- GenEval, Hugging Face model assets, Gurobi, DreamSim, LPIPS, HPSv2, ImageReward and other installed packages retain their own terms. The setup scripts obtain these separately on the VM. Model weights and licenses/credentials are not redistributed here.

Exact upstream revisions are in `reproduction/third_party/upstream.lock.json`; local source changes are documented in `reproduction/patches/`.
