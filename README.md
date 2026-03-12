# DAPO-Math

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/DAPOMath)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)

## Description

DAPO-Math is an environment for evaluating mathematical reasoning on competition-level problems from the DAPO-Math-17k dataset. The dataset was curated by ByteDance Seed and Tsinghua AIR as the training set for DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization), an open-source reinforcement learning system for large language models. Problems span algebra, geometry, number theory, and combinatorics, with integer ground-truth answers verified via rule-based matching.

## Capabilities

- Solving competition-level mathematics problems
- Step-by-step mathematical reasoning
- Producing precise numerical answers

## Compute Requirements

Minimal. No sandbox or code execution is used. The environment runs rule-based answer verification only.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), matching the original dataset license.

## Tasks

There are approximately 14,100 tasks in a single `train` split (English subset of the deduplicated DAPO-Math-17k dataset). Each task is a competition-level math problem with an integer answer.

## Reward Structure

Binary reward based on rule-based answer verification using the `math_verify` library (style: `rule-lighteval/MATH_v2`):

- **1.0** if the submitted answer is mathematically equivalent to the ground truth
- **0.0** otherwise

No LLM grader is used.

## Data

Problems are sourced from the [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset, using the deduplicated English subset provided by [open-r1/DAPO-Math-17k-Processed](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed). The dataset is loaded from HuggingFace at server startup.

## Tools

| Tool | Description |
|------|-------------|
| `answer` | Submit a final answer for evaluation against the ground truth. |

## Time Horizon

Single-turn. The agent receives a math problem and submits one answer.

## Environment Difficulty

These are competition-level math problems (olympiad-style). The DAPO system trained on this dataset achieved 50 points on AIME 2024 using Qwen2.5-32B.

## Other Environment Requirements

No external API keys or secrets are required. Grading is entirely rule-based.

## Safety

This environment poses minimal safety risk. Agents solve self-contained math problems with no access to external systems, file systems, or network resources.

## Citations

```bibtex
@article{yu2025dapo,
  title={DAPO: An Open-Source LLM Reinforcement Learning System at Scale},
  author={Yu, Qiying and Zhang, Zheng and Zhu, Ruofei and Yuan, Yufeng and Zuo, Xiaochen and Yue, Yu and Fan, Tiantian and Liu, Gaohong and Liu, Lingjun and Liu, Xin and Lin, Haibin and Lin, Zhiqi and Ma, Bole and Sheng, Guangming and Tong, Yuxuan and Zhang, Chi and Zhang, Mofan and Zhang, Wang and Zhu, Hang and Zhu, Jinhua and Chen, Jiaze and Chen, Jiangjie and Wang, Chengyi and Yu, Hongli and Dai, Weinan and Song, Yuxuan and Wei, Xiangpeng and Zhou, Hao and Liu, Jingjing and Ma, Wei-Ying and Zhang, Ya-Qin and Yan, Lin and Qiao, Mu and Wu, Yonghui and Wang, Mingxuan},
  journal={arXiv preprint arXiv:2503.14476},
  year={2025}
}
```
