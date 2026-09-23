# Combined Research Papers


> Combined from the five unique uploaded PDFs. The uploaded `base close and base foundation merged.pdf` was not repeated because it duplicates two papers already included.


---

# Paper 1: 2410.05695v2 ( base foundation)


**Source file:** `2410.05695v2 ( base foundation).pdf`

**Pages:** 33


## Page 1

Unlocking the Capabilities of Thought:
A Reasoning Boundary Framework to Quantify and
Optimize Chain-of-Thought
Qiguang Chen†
Libo Qin‡∗
Jiaqi Wang♢
Jinxuan Zhou‡
Wanxiang Che†∗
† Research Center for Social Computing and Information Retrieval
† Harbin Institute of Technology
‡ School of Computer Science and Engineering, Central South University
♢The Chinese University of Hong Kong
{qgchen,car}@ir.hit.edu.cn, lbqin@csu.edu.cn
Abstract
Chain-of-Thought (CoT) reasoning has emerged as a promising approach for
enhancing the performance of large language models (LLMs) on complex reasoning
tasks. Recently, a series of studies attempt to explain the mechanisms underlying
CoT, aiming to deepen the understanding of its efficacy. Nevertheless, the existing
research faces two major challenges: (1) a lack of quantitative metrics to assess
CoT capabilities and (2) a dearth of guidance on optimizing CoT performance.
Motivated by this, in this work, we introduce a novel reasoning boundary framework
(RBF) to address these challenges. To solve the lack of quantification, we first define
a reasoning boundary (RB) to quantify the upper-bound of CoT and establish a
combination law for RB, enabling a practical quantitative approach applicable to
various real-world CoT tasks. To address the lack of optimization, we propose three
categories of RBs. We further optimize these categories with combination laws
focused on RB promotion and reasoning path optimization for CoT improvement.
Through extensive experiments on 27 models and 5 tasks, the study validates the
existence and rationality of the proposed framework. Furthermore, it explains the
effectiveness of 10 CoT strategies and guides optimization from two perspectives.
We hope this work can provide a comprehensive understanding of the boundaries
and optimization strategies for reasoning in LLMs. Our code and data are available
at https://github.com/LightChen233/reasoning-boundary.
1
Introduction
In recent years, Large Language Models (LLMs) have demonstrated increasing capabilities and
applications across various tasks [Zhao et al., 2023, Chang et al., 2023, Pan et al., 2023, Qin
et al., 2024a]. Notably, advanced LLMs, such as GPT [Brown et al., 2020, OpenAI, 2022, 2023],
PaLM [Anil et al., 2023] and LlaMa [Touvron et al., 2023a,b, Meta, 2024] series have demonstrated
emergent capabilities, particularly like Chain-of-Thought (CoT) [Nye et al., 2022, Wei et al., 2022].
This methodology enables models to verbalize step-by-step reasoning, thereby enhancing prediction
accuracy by basing decisions on the logical rationale [Wei et al., 2022, Kojima et al., 2022, Hu et al.,
2024, Qin et al., 2023, Zhuang et al., 2023, Chen et al., 2024a].
Recently, some research in the literature has begun to investigate the mechanism of CoT to enhance
the understanding of its operational nature. To this end, Madaan et al. [2023] and Wang et al. [2023a]
∗Corresponding Author
38th Conference on Neural Information Processing Systems (NeurIPS 2024).
arXiv:2410.05695v2  [cs.CL]  28 Oct 2024


## Page 2

first give a qualitative boundary conclusion through a large number of experiments on the natural
language planning capability: The CoT is limited by the reasoning logic in the context demonstrations.
Bi et al. [2024] investigate these boundaries on the code planning capability, by training LLMs on
CoT samples of varying difficulties. It demonstrates LLMs are unable to learn or effectively manage
tasks that exceed a certain complexity upper-bound. To delve deeper into potential constraints of
CoT, Feng et al. [2024] develop a theoretical framework on the single-step calculation capability,
suggesting that there is an upper-bound of model performance dependent on the length of input in
single-step reasoning processes. Although existing research has made some progress, where the
boundaries of CoT lie and how these boundaries affect the performance of CoT are still unresolved
questions. Specifically, the existing work still faces two major challenges:
• Lacking quantification metrics for CoT: Current research primarily relies on qualitative
assessments of CoT performance, which leads to the absence of quantitative metrics. It hinders
the ability to objectively compare different CoT approaches and establish a definitive upper-bound
for CoT capabilities.
• Lacking optimization guidance for CoT: While current research prioritizes understanding
the mechanisms underlying CoT reasoning, there is a dearth of guidance on optimizing CoT
performance. This gap hinders the transformation of CoT research into actionable strategies for
enhancing model capabilities.
Motivated by this, in this work, we introduce a reasoning boundary framework (RBF) to thoroughly
examine and optimize the boundaries of current LLMs. Specifically, to address the quantification
challenge, we propose a new concept, named reasoning boundary (RB) to quantify the upper-bound
on task-specific reasoning complexity within a model. Furthermore, to explore more practical
scenarios, we present the combination law of RBs to generalize the RB for quantification in more real
and complex scenarios. To address the CoT optimization challenge, we propose and analyze three
reasoning boundary intervals, guiding optimization through improved RB and optimized reasoning
paths based on the combination law, which achieves state-of-the-art performance in our proposed
benchmark. We extensively validate the efficacy of our framework across 27 models and 5 tasks:
arithmetic computing, mathematical reasoning, multi-hop question answering, and multilingual
mathematical reasoning.
Our main contributions are as follows:
• To the best of our knowledge, this is the first work to propose a reasoning boundary framework
(RBF) to quantify the upper-bound of CoT. Furthermore, we establish the combination law of RB
as the weighted harmonic mean of fundamental RBs to address practical CoT tasks.
• To solve the lack of CoT optimization, we define three categories of RBs. Based on the
combination law and the nature of these RBs, we effectively improve the existing CoT strategies
by RB promotion and reasoning path optimization.
• We validate the existence and rationality of our framework on 27 models and 5 CoT tasks. Fur-
thermore, we explain the optimal performance from two optimization perspectives in numerous
CoT strategies. We consider both optimal perspectives and propose a minimum acceptable
reasoning path (MARP) prompting to achieve state-of-the-art performance.
2
Quantification Methodology
2.1
Reasoning Boundary
In order to quantify the capacity for complex reasoning in LLMs, we introduce an upper-bound
concept termed reasoning boundary (RB), which formally defines the degree of ease that an LLM can
handle within a specific reasoning process. In simpler terms, as shown in Figure 1 (a), RB reflects the
limit beyond which a model’s accuracy significantly degrades. Mathematically, RB is defined for a
model m and a task t as the maximum of problem difficulty d at which the model’s accuracy reaches
a predefined threshold K1:
BAcc=K1(t|m) = sup
d
{d|Acc(t|d, m) = K1},
(1)
where Acc(t|d, m) represents the accuracy of the model’s accuracy on task t with difficulty d.
Difficulty can be measured by factors like the number of reasoning steps or computational complexity.
For brevity, we denote RB as B(t|m) in subsequent sections.
2


## Page 3

Acc=90%
Acc=50%
399
Completely Feasible Reasoning 
Boundary (CFRB)
1,209,653
(1+2)*3 – 4 = ?
(1123+231)*332=?
𝓑Acc=90%
𝓑Acc=10%
Arithmetical Cal-
culation RB (𝓑(𝒂))
Mathematical 
Reasoning RB (𝓑(𝒂, 𝒑))
Weighted 
Harmonic Mean
Natural Language 
Planning RB (𝓑(𝒑))
Reasoning 
Boundary
(112+21)*3=?
𝓑Acc=50%
Acc=10%
5
Arithmetical Calculation + Planning → Mathematical Reasoning
1+2=?; 3*3=?; …
Step1: First you need to…
Step2: Then you need to…
Step 1: First you need to
calculate that Joan is 1+2=3
years old.
Step 2: Then you need to
calculate that Jack is 3*3=9
years old…
𝓑Acc≥90%
Partially Feasible Reasoning 
Boundary (PFRB)
𝓑10%<Acc<90%
Completely Infeasible Reasoning 
Boundary (CIRB)
𝓑Acc≤10%
(a) Reasoning Boundary (§2.1)
(b) Combination Law of Reasoning Boundary (§2.2)
(c) Categories of Reasoning Boundary (§2.3)
Figure 1: Overview of the introduced concepts.
Conclusion: The reasoning boundary for a model is defined by its ability to achieve a specific
accuracy for a given task difficulty.
2.2
Combination Law of Reasoning Boundary
In practical scenarios, models often require the integration of multiple capabilities to address a single
task effectively. To quantify how a large language model can be boosted by the cooperation of
multiple capabilities through the CoT mechanism, we introduce the “Combination Law of RB”, giving
a concrete formula of the upper-bound of the CoT. The law estimates the unified reasoning boundary
BAcc=K1(t1, t2, . . . , tn|m) for n tasks within a model m, which is formulated as:
BAcc=K1(t1, t2, . . . , tn|m) ≈
1
(n −1) Pn
i=1
Ni
BAcc=K1(ti|m)−bi
,
(2)
where BAcc=K1(ti|m) denotes the reasoning boundary of model m for task ti. Ni, and bi are scaling
factors, which are only affected by the related task. As shown in Figure 1 (b), Equation (2) provides a
mathematical formula to estimate the combined RBs from the independent ones, enabling deeper
insights into model behavior for intricate tasks. See Appendix A for detailed mathematical analysis.
Furthermore, the combination law for reasoning boundary demonstrates favorable theoretical prop-
erties, with broad applicability across diverse scenarios and flexibility in accommodating various
boundary segmentation methods. For detailed practical application, please refer to Appendix B.
Conclusion: The combination law of reasoning boundary satisfies the weighted harmonic
average of each basic reasoning boundary.
2.3
Categories of Reasoning Boundary
Furthermore, in order to guide the optimization of CoT and more convenient expression, as shown in
Figure 1 (c), we define the following three categories of RBs based on their empirical accuracy:
Completely Feasible Reasoning Boundary:
We define that the part with an accuracy greater
than 90% is a completely feasible reasoning boundary (CFRB = BAcc≥90%(t1, t2, . . . , tn|m)), which
means that LLMs can effectively grasp the performance of this part.
Completely Infeasible Reasoning Boundary: We believe that the part with an accuracy less than
10% is a completely infeasible reasoning boundary (CIRB = BAcc≤10%(t1, t2, . . . , tn|m)), which
means that the model can never effectively grasp the performance of this part.
Partially Feasible Reasoning Boundary: We define the RB in the rest part except CFRB and CIRB as
a partially feasible reasoning boundary (PFRB = B10%<Acc<90%(t1, t2, . . . , tn|m)), which requires
the model to repeat thinking or more clear information to solve the problem.
3


## Page 4

(a) Distribution of correct predictions
for x*y samples.
Correct
Incorrect
CFRB
PFRB
(b) Distribution of correct predictions
for nature language planning.
CIRB
x
y
0
500
1000
1500
2000
500
1000
1500
2000
0
x*y
=2.2e5
x*y
=2e6
(c) Distribution of correct predictions
for code planning.
The number of planning step (𝓑(p))
The number of planning step (𝓑(p))
Accuracy (%)
15
Accuracy (%)
1
3
5
7
9
11
100
80
60
40
20
0
13
15
1
3
5
7
9
11
100
80
60
40
20
0
13
Figure 2: Existence Verification for Reasoning Boundary.
We analyze the nature of these three categories of RB in detail (in Section 4.3), and further utilize
the combination law to optimize these three reasoning boundaries (in Section 5), so as to provide
effective suggestions and guidance to support future CoT optimization.
3
Experimental Setup
Benchmark Settings
To assess the reasoning boundaries of LLMs, we require a dataset rich in RB.
This necessitates tasks with evenly distributed complexities and reasoning steps that challenge the
models’ upper-bounds. To meet these requirements, we introduce BIGGSM, a new dataset offering
greater calculation complexity and longer reasoning chains. The detailed construction process for
BIGGSM is provided in Appendix C.
Model Settings
Except for model expansion experiments, all experiments are conducted on GPT-
3.5-Turbo. Following the setting of Wei et al. [2022], in our CoT experiment, all multi-step reasoning
tasks utilize three manually constructed demonstrations. In addition, for all the experiments, top-p is
selected from {0.95, 1}. Temperature is selected from [0, 1] and serves as the main error variable.
4
Empirical Analysis of Reasoning Boundary
4.1
Existence Verification for Reasoning Boundary
In this study, we investigate the hypothesis that an LLM exhibits varying levels of reasoning boundary
across various tasks. To this end, we will verify whether the model has widespread reasoning
boundary in various tasks in the following three tasks:
Basic Arithmetic Calculation
First, to investigate the existence of RB, we first examine basic
arithmetic operations (including addition, subtraction, multiplication, and division). As illustrated in
Figure 2 (a), the results reveal significant performance variations across three distinct regions. For
multiplication, accuracy surpasses 90% for results up to 2.2e5. Conversely, accuracy falls below 10%
for products exceeding 2e6. Similar presences of varying RBs are observed for other operations,
which verifies the existence of reasoning boundary in basic arithmetic calculation tasks. Further
results and implementation details are provided in Appendix D.
Nature Language Planning
We further investigate RB in natural language planning tasks for
mathematical reasoning. We prompt the model to generate plans and assess their accuracy through
manual evaluation. There is a strong correlation between the number of reasoning steps and LLMs’
performance in Figure 2 (b). When the model meets the question with fewer than 2 reasoning steps,
accuracy surpasses 90%. Conversely, when reasoning steps exceed 4, accuracy falls below 10%. This
finding suggests that there are also three different RB categories in natural language planning tasks.
Code Planning
For further extensive exploration, we further prompt LLMs by PAL [Gao et al.,
2023] to generate code-format plans and evaluate them by manual annotation. As shown in Figure 2
(c), the code planning task is similar to natural language planning, which is also an obvious division
and different categories of RBs. Notably, since code planning utilizes code for clearer logic and
reduced expression complexity, its planning accuracy surpasses that of natural language planning.
4


## Page 5

Maximum multiplication 
calculation value (𝓑(m))
0
5
10
15
20
0
2e4
4e4
6e4
8e4
1e5
The number of calculation step (𝓑(s))
(a) The combination law of different 
reasoning 
boundaries 
in 
complex 
calculation task.
Correct sample
Incorrect sample
CFRB
PFRB
CIRB
(b) The combination law of different 
reasoning boundaries in mathematical 
reasoning task.
The number of entity (𝓑(e))
The number of hop (𝓑(h))
(c) The combination law of different 
reasoning boundaries in multi-hop 
question-answering task.
12
10
8
6
4
2
0
10
20
30
40
The number of planning step (𝓑(p))
Maximum multiplication 
calculation value (𝓑(m))
0
5e4
1e5
1.5e5
2e5
2.5e5
3e5
4
1
7
10
13
16
Figure 3: Combination law verification of RB on different tasks. More verification results on other
tasks are shown in Figure 12.
4.2
Combination Law Verification on Different Tasks
Combination Law in Complex Arithmetic Calculation
Building on the proof of Equation (13),
we hypothesize that the combination law for RB in the complex arithmetic calculation is the harmonic
average of the arithmetic calculation RB and calculation planning RB. To verify this, we designed
an experiment focusing on formulas containing addition, subtraction, and multiplication, like “(1 +
2) ∗3 −4”. Since addition and subtraction complexities are assumed to be around 1e15 (as shown in
Figure 13), the arithmetic calculation RB primarily depends on the multiplication RB and calculation
planning RB. Therefore, as shown in Figure 3 (a), there are two obvious RB lines, namely BAcc=90%
and BAcc=10%, which are completely consistent with the combination law of these basic RB based
on the Equation (2). Besides, these two lines also clearly divide the RBs into three categories.
Combination Law in Mathematical Reasoning
Inspired by Tan [2023b], Xiao and Liu [2024],
we posit that the natural language mathematical CoT task is determined by two sub-tasks: step
planning task and step calculation task for global logic planning and local mathematical calculation.
Furthermore, each model output step requires a single basic operation, resulting in a step calculation
boundary close to the maximum number of multiplications, denoted by B(c) ≈B(m). Formally,
with step planning RB denoted by (B(p)) and the step calculation RB by (B(c)), then the combined
RB satisfies the following law:
BCoT(c, p) =
1
N1
(B(c)−b1) +
N2
(B(p)−b2)
.
(3)
As illustrated in Figure 3 (b), the actual performance distribution of RB (including BAcc=90%
and BAcc=10%) in natural language mathematical reasoning task fully aligns with the proposed
combination law in Equation (3). Additionally, there are also obviously three RBs in Figure 3 (b).
Combination Law in Multi-hop Reasoning
Beyond the realm of mathematics, we further extend
our exploration of the combination law to the field of multi-hop question answering. Specifically,
we validate our law on HotpotQA [Yang et al., 2018], where we define the reasoning boundary as
the combination of global hop-planning RB and local knowledge entity reasoning RB. As shown
in Figure 3 (c), BAcc=90% and BAcc=10% also satisfy the weighted harmonic mean of these two
sub-reasoning boundaries. It is also proved that, in addition to math-related tasks, multi-hop question
answering also satisfies our proposed combined law and also exhibits three distinct RBs. We will
describe in detail how to calculate the combination law on multi-hop reasoning in Appendix E.
4.3
Nature Analysis for different Reasoning Boundary
According to the definition of different RBs, we have divided the problem into three parts for LLMs.
In this section, we will verify whether the defined RB adheres to the intrinsic nature of the model
itself. We will discuss the natures of these RBs in detail:
CFRB means complete mastery of the model even without demonstration. According to the
definition, we assume that a question within CFRB implies a comprehensive understanding of the
associated issue for a certain LLM. To verify this, following Zhang et al. [2022] and Wei et al. [2022],
we formulate a mathematical request and generate chain-of-thought rationale and answer through
5


## Page 6

Accuracy (%)
(a) The
accuracy distribution of
generated rationales based on Auto-
CoT and Zero-CoT.
89.7 
89.7 
90.7 
91.6 
53.0 
54.8 
56.5 
57.4 
9.5 
9.5 
9.5 
9.5 
0
1
2
3
4
5
0
10
20
30
40
50
60
70
80
90
100
#1
#3
#4
#5
The size of self-consistency
Accuracy (%)
(b) Model self-consistency integrated performance in dif-
ferent Reasoning Boundary areas.
Δ Accuracy (%)
ΔCFRB
ΔPFRB
ΔCIRB
CFRB
PFRB
CIRB
Reasoning Boundary Type
84.1 
54.9
12.7
5
20
35
50
65
80
95
CFRB
PFRB CIRB
77.1 
51.1 
50.0 
45
55
65
75
85
Accuracy (%)
CFRB
PFRB
CIRB
(c) The accuracy (top) and quantity
(bottom) distribution of synthetic
samples from Synthetic-CoT.
CFRB
PCFRB
IFRB
Error
65%
25
1%
9%
Figure 4: Nature analysis at different reasoning boundaries.
zero-shot prompting without any demonstration. As shown in Figure 4 (a), it still achieves 29.2%
improvement in CFRB on generating the correct rationale compared to other RBs. This also proves
that the model can indeed master tasks well on the questions in CFRB.
PFRB means moderate confidence in its solution and needs consensus building process. To
gauge the level of performance and confidence, we draw parallels to human decision-making, where
moderate confidence often necessitates multiple times of consensus building. Inspired by this, we
investigate it on Self-Consistency [Wang et al., 2022], which integrates results from various
reasoning answers to reach a conclusive answer. Figure 4 (b) demonstrates that as the integration of
reasoning paths increases, the accuracy improves significantly within PFRB compared with other RBs.
This suggests that within PFRB, the LLM exhibits moderate confidence in solving problems, which
needs multiple consensus building.
CIRB exhibits poor reasoning performance even with consensus building. As illustrated in
Figure 4 (a), questions in CIRB display extremely low accuracy (around 9.5%). And the model shows
consistently poor performance and no improvement on Self-consistency in this boundary in
Figure 4. It signifies that the model exhibits poor reasoning performance.
LLM has self-awareness of its own RBs. In parallel, a natural question arises: Is the model capable
of discerning its inherent RBs? To investigate this, we employ the Synthetic-CoT [Shao et al.,
2023] to prompt LLM to generate CoT data. As depicted in Figure 4 (c), the results demonstrated
that there are over 65% of generated samples within CFRB, which achieves a much higher percentage
and performance than other RBs. This suggests that LLMs possess an intrinsic understanding of their
RBs and constraints to generate the task they grasp, indicative of a potential for self-assessment.
Takeaways: (1) Reasoning boundary (RB) and the combination law of RB are both widespread
across a series of tasks. (2) Different categories of RB can reflect the corresponding performance,
and the model can also have a self-understanding of its own RB.
5
RB-based CoT Optimization
5.1
How can we improve CoT by optimizing RB?
Based on our framework, the reasoning boundary limits the performance of the model. The simplest
approach to improve CoT is to optimize the step calculation RB B(c) to promote the value of
RB. Specifically, Tool-Usage [Paranjape et al., 2023] and Program-of-Thought (PoT) [Chen et al.,
2024b] have shown significant success in CoT optimization. We explain the rationale behind their
effectiveness, why PoT consistently outperforms direct Tool Usage [Yao et al., 2023, Chen et al.,
2023], and take them as examples to demonstrate how to improve CoT by promoting RB.
Tool Usage can boost the value of RB for an LLM. When the model uses tools [Paranjape et al.,
2023], we can simply think that the model can perform calculations with infinite precision, so that the
RB of mathematical calculations tends to infinity, viz B(c) →+∞. It is obvious that the combined
6


## Page 7

Model
BIGGSM
Acc. (↑)
Input Token (↓)
Output Token (↓)
CoT
57.00 ±0.93
780.43
96.76 ±3.22
RB-Optimized Methods
Tool Usage
71.64 ±0.66
688.43
129.53 ±3.82
PoT
78.25 ±1.09
657.43
78.25 ±1.09
Reasoning-Path-Optimized Methods
Least-to-most
58.25 ±3.28
679.59
176.09 ±15.22
Complex-CoT
59.78 ±0.60
1111.43
131.82 ±1.91
CoT+MARP
64.37 ±2.24
614.43
95.12 ±0.77
PoT+MARP
80.55 ±2.40
576.43
76.34 ±2.84
Table 1: Main experimental results on GPT-3.5-Turbo.
Results on different benchmarks are shown in Table 2.
𝓑(c, p)
𝓑"#$%
𝓑"&$%
𝓑'($%
𝓑')$%
0.8
0.6
0.4
0.2
0
𝓑!"#
𝓑#""$
𝓑%"#
Practical value 
Theoretical interval 
𝓑!"#
𝓑#""$
Figure 5: Analysis of the impact of Tool-
Usage and PoT on reasoning boundary
B(c, p).
RB of the model can be calculated as:
BTool(c, p) =
lim
B(c)→+∞
1
N1
(B(c)−b1) +
N2
(B(p)−b2)
= B(p) −b2
N2
.
(4)
Easy to get, BTool(c, p) > BCoT(c, p), this shows that Tool Usage can improve the boundary of
reasoning. This explains why Tool Usage can have better performance than vanilla CoT (as shown in
Table 1). Furthermore, as shown in Figure 5, the distribution of theoretical RB and the actual one
almost perfectly coincide. This also demonstrates the reliability and applicability of our theory.
Program-of-Thought can further enhance the value of LLM’s RB. Equation (4) reveals that an
LLM’s RB hinges entirely on its planning capability. Since natural language can be verbose, it hinders
the planning capability of LLM [Gao et al., 2023, Hu et al., 2023, Puerto et al., 2024, Chen et al.,
2024b]. PoT [Chen et al., 2023] offers a clearer representation of logic using code, allowing for clearer
planning (as shown in Figure 2 (b, c)). This leads to finer-grained planning reasoning B∗(p) > B(p).
Then the PoT reasoning boundary BPoT(c, p) > BTool(c, p), aligning with the observed performance
gains of PoT over Tool Usage (see Table 1). Furthermore, Figure 5 visually demonstrates that PoT’s
theoretical and practical reasoning boundaries consistently outperform Tool Usage. This reinforces
the theoretical advantage of PoT and its empirical effectiveness.
5.2
How can we improve CoT based on a certain RB?
Enhancing RB is crucial for optimizing CoT, but requires changes to the model or its reasoning
architecture to be effective. Therefore, we need to consider how to optimize the reasoning path so
that the difficulty satisfies the RB (d∗= BAcc=K1) instead of the original RB (d = BAcc=K2), where
K2 < K1. According to Equation (3), B is affected by both arithmetical RB and planning RB. Given
B, we consider optimizing reasoning ability from the following two strategies as examples 2:
Complex CoT (CCoT): By increasing the boundary of planning to reduce the pressure of single-step
calculation, reduce the arithmetical RB, and then get smaller d; However, it introduces more planning
steps, which adds the planning pressure. As shown in Figure 6, the model performance first increases
and then decreases with the increasing number of CCoT steps.
Least-to-Most (LtM): By dividing multiple sub-questions to reduce the pressure of local planning
within a sub-question, reduce the boundary of local planning, and then get smaller d. However,
even though it can release local planning pressure (as demonstrated in Figure 7), this approach
simultaneously intensifies global planning pressure by generating an excessive number of sub-
questions (as depicted in Figure 15).
Limitation:
(1) CCoT needs to keep balance in the number of reasoning steps and calculation
pressure. (2) Although the pressure of local planning has been reduced, LtM has not effectively
reduced the pressure of global planning, nor the pressure of optimization calculations.
Minimum acceptable reasoning paths prompting can further achieve better CoT within a spe-
cific RB.
To address the aforementioned two issues, we proposed Minimum Acceptable Reasoning
2See detailed analysis for these two strategies in Appendix F.
7


## Page 8

text-davince-002
text-davince-003
GPT3.5
Manual Complex CoT
Auto-Generated Complex CoT
GPT4
10
20
30
40
50
60
70
80
90
100
Accuracy (%)
The number of steps
0
1
2
3
4
5
6
90
91
92
93
94
95
The number of steps
Accuracy (%)
0
1
2
3
4
5
6
Figure 6: Correlation between the number of steps
and performance of Complex-CoT on GSM8K
(left) and SingleEq (right). See Appendix G for
more meta-analysis results.
66.7 
51.9 
42.5 
42.4 
30.6 
69.4 
65.8 
45.2 
44.1 
36.1 
0
10
20
30
40
50
60
70
80
0~6e4
6e4~1.2e5
1.2e5~1.8e5
2.4e5~3e5
3.6e5~4.2e5
Chain-of-Thought Prompting
Least-to-Most Prompting
Maximum multiplication calculation value (𝓑(m))
Accuracy
Figure 7: The performance distribution of
Least-to-Most prompting on different calcula-
tion amounts.
Paths (MARP). Our first objective is to alleviate the computational burden of the model. We achieve
this by introducing instructions that set an upper limit on its single-step computational capacity,
thereby optimizing the boundary of its computational reasoning. Secondly, we aim to enhance the
model’s acceptability. Within the calculation and planning boundary, we increase the amount of
computation performed in each step in demonstrations as much as possible while simultaneously
reducing the number of global planning steps, which effectively mitigates planning pressure. As
shown in Table 1, MARP demonstrably improves model performance and effectively reduces the
token consumption. By maximizing operations per step, MARP leads to a more streamlined and
efficient problem-solving process. Detailed descriptions of this strategy are shown in Appendix G.3.
Takeaways: (1) Tool-Usage and PoT can be utilized to optimize CoT by the calculation and
planning reasoning boundary optimization. (2) MARP can well lessen planning and calcula-
tion pressure by problem optimization in certain RB (3) Users can effectively optimize CoT
performance by optimizing the reasoning boundary and the problem.
6
Expansion Verification & Exploration
RB can be extended to various models.
To extend our mechanism’s applicability, we verify
the mechanism on 25 diverse models (details in Table 3). As shown in Figure 8 (a), we observe a
positive correlation between reasoning boundary and model accuracy on mathematical benchmarks.
Moreover, the models that use mathematical data such as MathInstruct for SFT, often have interesting
outliers that are different from the general LLMs’ area, but they also satisfy a positive correlation with
5
20
35
50
65
80
95
0
0.2
0.4
0.6
0.8
5
20
35
50
65
80
95
5
20
35
50
65
80
95
0
0.2
0.4
0.6
0.8
𝓑!""#$%%
𝓑!""'(%%
Accuracy (%)
Accuracy (%)
GSM8k
MATH
BigGSM
GSM8k
MATH
BigGSM
Math LLM
General LLM
Closed LLM
…
0.106 0.108
0.102 0.104
0
0.1
Open LLM
𝓑!""'(%%
Accuracy (%)
GSM8k
MATH
BigGSM
Math LLM
(a) Correlation between the values of CIRB 
𝓑)**#+,% for different general LLMs and 
performance on real benchmarks.
(b) Correlation between the values of CIRB 
𝓑)**#+,% for different math LLMs and 
performance on real benchmarks.
(b) Correlation between the values of CFRB 
𝓑)**'-,%for different closed and open
LLMs and performance on real benchmarks.
Figure 8: Correlation between the values of RB for different models and performance on real
benchmarks. See Appendix H for more empirical details.
8


## Page 9

LLaMA-2
Code-LLaMA
Parameters (B)
0.00
0.02
0.04
0.06
0.08
0.10
0.12
0.14
0
10
20
30
40
50
60
70
80
LLaMA
𝓑!""'(%%
Figure 9: Scaling law correlation between
model parameters and CIRB.
Language Performance (𝓑(l))
The number of planning step (𝓑(p))
Maximum multiplication 
calculation value (𝓑(m))
80
70
60
50
40
30
10
20
00
100000
200000
300000
400000
500000
0
1
2
3
4
5
6
7
8
Figure 10: Different boundaries on MGSM.
our RBs (as shown in Figure 8 (b)), which helps determine if the model underwent mathematically
targeted training.
However, as shown in Figure 8 (c), we find some interesting phenomena. For example, the main
difference between the current open-source model and the closed-source model is still CFRB. Except
for the closed source model, the CFRB of all models is 0. It shows the potential and the direction of
the model optimization. Furthermore, a scaling law of RB can also emerge (as shown in Figure 9):
reasoning boundary increased with model parameter count and data quality.
RB can be extended to more tasks. To assess the RB in more tasks, we evaluate them on a
multilingual mathematical reasoning task. Inspired by Qin et al. [2024b], we hypothesize that
multilingual RB, assessed through direct answer accuracy across different languages, mathematical
computation RB, represented by the maximum product result, and reasoning planning RB, indicated
by the planning steps, are orthogonal dimensions of performance. We propose that these RBs can be
effectively combined using a weighted harmonic mean. As illustrated in Figure 10 confirms that the
combined RB maintains the expected three different RBs. Detailed implementation description is
shown in Appendix I.
7
Related Work
In this section, we review recent literature related to Chain-of-Thought (CoT) prompting, focusing on
theoretical and empirical investigations. Madaan et al. [2023], Wang et al. [2023a], Saparov and He
[2023], He-Yueya et al. [2023], Zhang et al. [2024], Wang et al. [2024] and Prystawski et al. [2024]
qualitatively show that the LLMs learn the reasoning chain based on the demonstrations in the context.
Besides, Lampinen et al. [2022] and Tan [2023a] find a causal link between generated intermediate
steps and the final answers during a series of qualitative experiments. Wang et al. [2023c], Hanna et al.
[2024] and Dutta et al. [2024] study neural substructure within the LLMs, embodying CoT reasoning
from a white-box mechanism perspective, demonstrating that LLMs deploy multiple parallel answer
generation paths internally.
Recently, a large amount of work has demonstrated the upper-bounds and limitations of LLM in
various CoT tasks [Qin et al., 2023, Imani et al., 2023, Huang et al., 2024, Sprague et al., 2024]. Bi
et al. [2024] investigate these bounds on planning capability in code generation by training LLM
on CoT samples of varying difficulties. Their findings suggest that LLMs have a limited capacity
to learn or manage tasks exceeding a certain complexity threshold. Further understanding of the
CoT upper-bound, Merrill and Sabharwal [2023], Li et al. [2023] and Feng et al. [2024] analyze
single-step arithmetic capability, which suggests an upper bound on model performance related to
input length in single-step reasoning processes.
Despite advancements in CoT explanation for LLMs, significant challenges remain, including the
absence of quantifiable metrics for CoT’s upper-bounds and the deficiency in optimization guidelines.
To tackle this, we propose a reasoning boundaries framework (RBF) to systematically quantify
and optimize various CoT approaches. This framework offers a transferable and user-friendly
9


## Page 10

The number of planning step (𝓑(p))
Maximum multiplication 
calculation value (𝓑(m))
0
5e4
1e5
1.5e5
2e5
2.5e5
3e5
4
1
7
10
13
16
(a) GPT-3.5-turbo
The number of planning step (𝓑(p))
Maximum multiplication 
calculation value (𝓑(m))
0
5e4
1e5
1.5e5
2e5
2.5e5
3e5
4
1
7
10
13
16
(c) O1-preview
The number of planning step (𝓑(p))
Maximum multiplication 
calculation value (𝓑(m))
0
5e4
1e5
1.5e5
2e5
2.5e5
3e5
4
1
7
10
13
16
(b) GPT-4o
Correct sample
Incorrect sample
CFRB
PFRB
CIRB
Figure 11: Combination law verification of reasoning boundaries on GPT-series models.
methodology to enhance model performance from a mechanistic perspective. We anticipate that it
will furnish systematic insights for ongoing research and inform future developments in the field.
8
Discussion
Discussion on the Boundaries Improvements
Furthermore, in order to better understand the best
existing LLMs, we utilize RBF to test the current most advanced GPT-series models. As shown in
Figure 11, all reasoning boundaries improve a lot compared to the last version which also achieves
performance enhancement. Notably, the CFRB increases slightly compared with the improvement of
CIRB between GPT-3.5 and GPT-4o. But o1 significantly improves the CFRB. Furthermore, as shown
in Figure 14 in Appendix, o1 shows extremely significant improvements on CFRB, which is almost
three times of other models. We attribute it to the fact that the advanced Reinforce-Learning and
Inference Scaling strategies play a key role in improving this part of the ability compared with the
normal improvements in CFRB, which might trigger more in-depth research.
Broader impacts.
Our framework is the first work to quantify the reasoning upper-bound of LLMs.
This enables the explanation for a huge part of the valid CoT framework. We hope that our work can
provide new insights and more systematic guidance for future interpretability analysis of CoT. For
social impact, this work may have a certain impact on the controllable and explainable AGI.
Limitations & Future.
Due to the cost and time constraints, this work does not discuss the
complex relationships such as causal conditions among the basic RBs. In addition, evaluating the
robustness and applicability of CoT reasoning boundaries-related techniques in dynamic scenarios
will be crucial for future research.
9
Conclusion
This study introduces a novel reasoning boundaries framework (RBF) to quantify and optimize the
limitations of LLMs in CoT tasks. Specifically, we propose the concept of reasoning boundaries
(RBs) and the combination law of RBs in more complex scenarios for quantitative metrics. We
further introduce three categories of RB for CoT optimizations. The framework is validated through
extensive experiments across 27 models and 5 tasks. Furthermore, we improve the CoT in both RB
and question optimization perspectives to achieve state-of-the-art performance in BIGGSM. We
hope that this framework paves the way for further research on understanding and enhancing LLMs’
reasoning capabilities.
Acknowledgments
This work was supported by the National Natural Science Foundation of China (NSFC) via grant
62236004, 62441603, 62476073 and 62306342. This work was also sponsored by the Excellent Young
Scientists Fund in Hunan Province (2024JJ4070), the Science and Technology Innovation Program of
Hunan Province under Grant 2024RC3024, and the CCF-Zhipu.AI Large Model Innovation Fund.
10


## Page 11

References
Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos,
Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv
preprint arXiv:2305.10403, 2023.
Anthropic.
The claude 3 model family:
Opus, sonnet, haiku.
2024.
URL https:
//www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_
Card_Claude_3.pdf.
Zhen Bi, Ningyu Zhang, Yinuo Jiang, Shumin Deng, Guozhou Zheng, and Huajun Chen. When do
program-of-thought works for reasoning? In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 38, pages 17691–17699, 2024.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan
Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM
Transactions on Intelligent Systems and Technology, 2023.
Qiguang Chen, Libo Qin, Jin Zhang, Zhi Chen, Xiao Xu, and Wanxiang Che. M3CoT: A novel
benchmark for multi-domain multi-step multi-modal chain-of-thought. In Lun-Wei Ku, Andre
Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 8199–8221, Bangkok, Thailand,
August 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.446.
URL https://aclanthology.org/2024.acl-long.446.
Weize Chen, Chenfei Yuan, Jiarui Yuan, Yusheng Su, Chen Qian, Cheng Yang, Ruobing Xie, Zhiyuan
Liu, and Maosong Sun. Beyond natural language: Llms leveraging alternative formats for enhanced
reasoning and communication. arXiv preprint arXiv:2402.18439, 2024b.
Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting:
Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine
Learning Research, 2023.
Daixuan Cheng, Shaohan Huang, and Furu Wei. Adapting large language models via reading
comprehension. In The Twelfth International Conference on Learning Representations, 2024. URL
https://openreview.net/forum?id=y886UXPEZ0.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve
math word problems. arXiv preprint arXiv:2110.14168, 2021.
Subhabrata Dutta, Joykirat Singh, Soumen Chakrabarti, and Tanmoy Chakraborty. How to think
step-by-step: A mechanistic understanding of chain-of-thought reasoning.
arXiv preprint
arXiv:2402.18312, 2024.
Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing
the mystery behind chain of thought: a theoretical perspective. Advances in Neural Information
Processing Systems, 36, 2024.
Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting
for multi-step reasoning. In The Eleventh International Conference on Learning Representations,
2023. URL https://openreview.net/forum?id=yf1icZHC-l9.
Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and
Graham Neubig. PAL: Program-aided language models. In Andreas Krause, Emma Brunskill,
Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of
the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine
Learning Research, pages 10764–10799. PMLR, 23–29 Jul 2023. URL https://proceedings.
mlr.press/v202/gao23f.html.
11


## Page 12

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did Aristotle
Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies. Transactions
of the Association for Computational Linguistics, 9:346–361, 04 2021. ISSN 2307-387X. doi:
10.1162/tacl_a_00370. URL https://doi.org/10.1162/tacl_a_00370.
Michael Hanna, Ollie Liu, and Alexandre Variengien. How does gpt-2 compute greater-than?: Inter-
preting mathematical abilities in a pre-trained language model. Advances in Neural Information
Processing Systems, 36, 2024.
Joy He-Yueya, Gabriel Poesia, Rose Wang, and Noah Goodman. Solving math word problems
by combining language models with symbolic solvers. In The 3rd Workshop on Mathematical
Reasoning and AI at NeurIPS’23, 2023.
Mengkang Hu, Yao Mu, Xinmiao Chelsey Yu, Mingyu Ding, Shiguang Wu, Wenqi Shao, Qiguang
Chen, Bin Wang, Yu Qiao, and Ping Luo. Tree-planner: Efficient close-loop task planning with
large language models. In The Twelfth International Conference on Learning Representations,
2024.
Yi Hu, Haotong Yang, Zhouchen Lin, and Muhan Zhang. Code prompting: a neural symbolic method
for complex reasoning in large language models. arXiv preprint arXiv:2305.18507, 2023.
Haoyang Huang, Tianyi Tang, Dongdong Zhang, Wayne Xin Zhao, Ting Song, Yan Xia, and Furu
Wei. Not all languages are created equal in llms: Improving multilingual capability by cross-
lingual-thought prompting. In Findings of the Association for Computational Linguistics: EMNLP
2023, pages 12365–12394, 2023.
Jen-tse Huang, Eric John Li, Man Ho Lam, Tian Liang, Wenxuan Wang, Youliang Yuan, Wenxiang
Jiao, Xing Wang, Zhaopeng Tu, and Michael R Lyu. How far are we on the decision-making of llms?
evaluating llms’ gaming ability in multi-agent environments. arXiv preprint arXiv:2403.11807,
2024.
Shima Imani, Liang Du, and Harsh Shrivastava. Mathprompter: Mathematical reasoning using large
language models. In Proceedings of the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 5: Industry Track), pages 37–42, 2023.
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot,
Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al.
Mistral 7b. arXiv preprint arXiv:2310.06825, 2023a.
Song Jiang, Zahra Shakeri, Aaron Chan, Maziar Sanjabi, Hamed Firooz, Yinglong Xia, Bugra
Akyildiz, Yizhou Sun, Jinchao Li, Qifan Wang, et al. Resprompt: Residual connection prompting
advances multi-step reasoning in large language models. arXiv preprint arXiv:2310.04743, 2023b.
Mingyu Jin, Qinkai Yu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, Mengnan Du, et al.
The impact of reasoning step length on large language models. arXiv preprint arXiv:2401.04925,
2024.
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
language models are zero-shot reasoners. Advances in neural information processing systems, 35:
22199–22213, 2022.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating
Systems Principles, 2023.
Andrew Lampinen, Ishita Dasgupta, Stephanie Chan, Kory Mathewson, Mh Tessler, Antonia Creswell,
James McClelland, Jane Wang, and Felix Hill. Can language models learn from explanations
in context?
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Findings of the
Association for Computational Linguistics: EMNLP 2022, pages 537–563, Abu Dhabi, United
Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/
2022.findings-emnlp.38. URL https://aclanthology.org/2022.findings-emnlp.38.
12


## Page 13

Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transform-
ers to solve inherently serial problems. In The Twelfth International Conference on Learning
Representations, 2023.
Aman Madaan, Katherine Hermann, and Amir Yazdanbakhsh. What makes chain-of-thought prompt-
ing effective? a counterfactual study. In Houda Bouamor, Juan Pino, and Kalika Bali, editors,
Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1448–1535,
Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.
findings-emnlp.101. URL https://aclanthology.org/2023.findings-emnlp.101.
William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought.
In The Twelfth International Conference on Learning Representations, 2023.
Meta. Llama 3. 2024.
Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David
Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work:
Scratchpads for intermediate computation with language models. In Deep Learning for Code
Workshop, 2022.
OpenAI. Introducing chatgpt. 2022.
OpenAI. Gpt-4 technical report, 2023.
Wenbo Pan, Qiguang Chen, Xiao Xu, Wanxiang Che, and Libo Qin. A preliminary evaluation of
chatgpt for zero-shot dialogue understanding. arXiv preprint arXiv:2304.04256, 2023.
Bhargavi Paranjape, Scott Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, and
Marco Tulio Ribeiro. Art: Automatic multi-step reasoning and tool-use for large language models.
arXiv preprint arXiv:2303.09014, 2023.
Ben Prystawski, Michael Li, and Noah Goodman. Why think step by step? reasoning emerges from
the locality of experience. Advances in Neural Information Processing Systems, 36, 2024.
Haritz Puerto, Martin Tutek, Somak Aditya, Xiaodan Zhu, and Iryna Gurevych. Code prompting
elicits conditional reasoning abilities in text+ code llms. arXiv preprint arXiv:2401.10065, 2024.
Libo Qin, Qiguang Chen, Fuxuan Wei, Shijue Huang, and Wanxiang Che. Cross-lingual prompting:
Improving zero-shot chain-of-thought reasoning across languages. In Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing, pages 2695–2709, 2023.
Libo Qin, Qiguang Chen, Xiachong Feng, Yang Wu, Yongheng Zhang, Yinghui Li, Min Li, Wanxiang
Che, and Philip S Yu. Large language models meet nlp: A survey. arXiv preprint arXiv:2405.12819,
2024a.
Libo Qin, Qiguang Chen, Yuhang Zhou, Zhi Chen, Yinghui Li, Lizi Liao, Min Li, Wanxiang Che, and
Philip S Yu. Multilingual large language model: A survey of resources, taxonomy and frontiers.
arXiv preprint arXiv:2404.04925, 2024b.
Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi
Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code.
arXiv preprint arXiv:2308.12950, 2023.
Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis
of chain-of-thought. In The Eleventh International Conference on Learning Representations, 2023.
URL https://openreview.net/forum?id=qFVVBzXxR2V.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. Syn-
thetic prompting: Generating chain-of-thought demonstrations for large language models. In
International Conference on Machine Learning, pages 30706–30775. PMLR, 2023.
Kashun Shum, Shizhe Diao, and Tong Zhang. Automatic prompt augmentation and selection with
chain-of-thought from labeled data. In Findings of the Association for Computational Linguistics:
EMNLP 2023, pages 12113–12139, 2023.
13


## Page 14

Zayne Rea Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. MuSR: Testing the
limits of chain-of-thought with multistep soft reasoning. In The Twelfth International Conference
on Learning Representations, 2024. URL https://openreview.net/forum?id=jenyYQzue1.
Jiashuo Sun, Yi Luo, Yeyun Gong, Chen Lin, Yelong Shen, Jian Guo, and Nan Duan. Enhancing
chain-of-thoughts prompting with iterative bootstrapping in large language models. arXiv preprint
arXiv:2304.11657, 2023.
Juanhe (TJ) Tan. Causal abstraction for chain-of-thought reasoning in arithmetic word problems. In
Yonatan Belinkov, Sophie Hao, Jaap Jumelet, Najoung Kim, Arya McCarthy, and Hosein Mohebbi,
editors, Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural
Networks for NLP, pages 155–168, Singapore, December 2023a. Association for Computational
Linguistics. doi: 10.18653/v1/2023.blackboxnlp-1.12. URL https://aclanthology.org/
2023.blackboxnlp-1.12.
Juanhe TJ Tan. Causal abstraction for chain-of-thought reasoning in arithmetic word problems. In
Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for
NLP, pages 155–168, 2023b.
Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu
Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable
multimodal models. arXiv preprint arXiv:2312.11805, 2023.
Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Git-
man.
Openmathinstruct-1: A 1.8 million math instruction tuning dataset.
arXiv preprint
arXiv:2402.10176, 2024.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée
Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and
efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation
and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.
Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the
planning abilities of large language models-a critical investigation. Advances in Neural Information
Processing Systems, 36:75993–76005, 2023.
Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan
Sun. Towards understanding chain-of-thought prompting: An empirical study of what matters.
In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 2717–2739, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi:
10.18653/v1/2023.acl-long.153. URL https://aclanthology.org/2023.acl-long.153.
Haoyu Wang, Hongming Zhang, Yueguan Wang, Yuqian Deng, Muhao Chen, and Dan Roth. Are all
steps equally important? benchmarking essentiality detection in event processes. In Proceedings
of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4048–4056,
2023b.
Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong, and Yangqiu Song. Rethinking the bounds of
llm reasoning: Are multi-agent discussions the key? arXiv preprint arXiv:2402.18272, 2024.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdh-
ery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models.
arXiv preprint arXiv:2203.11171, 2022.
Yiqun Wang, Sile Hu, Yonggang Zhang, Xiang Tian, Xuesong Liu, Yaowu Chen, Xu Shen, and
Jieping Ye. How large language models implement chain-of-thought? 2023c.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in
neural information processing systems, 35:24824–24837, 2022.
14


## Page 15

Changnan Xiao and Bing Liu. A theory for length generalization in learning to reason. arXiv preprint
arXiv:2404.00560, 2024.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing, pages 2369–2380, 2018.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan
Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International
Conference on Learning Representations, 2023.
Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen.
Mammoth: Building math generalist models through hybrid instruction tuning. In The Twelfth
International Conference on Learning Representations, 2023.
Yufeng Zhang, Xuepeng Wang, Lingxiang Wu, and Jinqiao Wang. Pattern-aware chain-of-thought
prompting in large language models. arXiv preprint arXiv:2404.14812, 2024.
Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in
large language models. In The Eleventh International Conference on Learning Representations,
2022.
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min,
Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv
preprint arXiv:2303.18223, 2023.
Ziyu Zhuang, Qiguang Chen, Longxuan Ma, Mingda Li, Yi Han, Yushan Qian, Haopeng Bai, Zixian
Feng, Weinan Zhang, and Ting Liu. Through the lens of core competency: Survey on evaluation of
large language models. arXiv preprint arXiv:2308.07902, 2023.
15


## Page 16

Appendix
A
Mathematical Analysis & Proof
A.1
Definitions & Assumptions
In order to further quantify and analyze the combination law of RB, we will make the following
definitions and assumptions about the properties of RB:
Definition 1 If a basic RB in the combined RB takes infinity, then the combined RB only depends on
the remaining basic RBs.
That is, the model is no longer limited by a certain ability when solving tasks, and only needs to focus
on other ability shortcomings. Formally, combination law satisfies that3:
B(t1, t2, . . . , tn|m)=B(+∞, t2, . . . , tn|m)+B(t1, +∞, . . . , tn|m)+· · ·+B(t1, t2, . . . , +∞|m)
(5)
= B(t2, t3 . . . , tn|m)+B(t1, t3, . . . , tn|m)+· · ·+B(t1, t2, . . . , tn−1|m)
(6)
= (n −1)
n
X
i=1
B(+∞, . . . , +∞, ti, +∞, . . . , +∞|m)
(7)
Definition 2 If all basic RBs are infinite, it means that the model is omnipotent, and the combined
RB is also infinite.
Formally, the combination law satisfies that:
B(+∞, +∞, . . . , +∞|m) = +∞
(8)
Assumption 3 The combination law function is continuously differentiable everywhere.
Assumption 4 All basic reasoning boundary for combined reasoning boundary are mutually inde-
pendent.
A.2
The Proof of Combination Law
Based on the above definitions and assumptions, we need to prove that the combination law is a
combined RB and is the weighted harmonic average of two basic RBs.
Proof. Since Taylor expansion cannot be performed on infinity and the function needs to converge,
we set ti =
1
xi + bi and
1
B(t1,t2,...,tn|m) = B∗(x1, x2, . . . , xn|m). Then following Equation (7), we
can get the B∗(x1, x2, . . . , xn|m) as:
B∗(x1, x2, . . . , xn|m) = (n −1)
n
X
i=1
B∗(0, . . . , xi, . . . , 0|m).
(9)
According to the Taylor expansion formula, we expand this formula at xi →ki, we can get:
B∗(x1, x2, . . . , xn|m) = (n −1)
n
X
i=1
+∞
X
j=1
Nij(xi −ki)j
(10)
= (n −1)
n
X
i=1
Ni1(xi −ki) + O(xi)
(11)
≈(n −1)
n
X
i=1
Ni1(xi −ki),
(12)
3The basic form B(ti|m) and the combined form B(+∞, . . . , +∞, ti, +∞, . . . , +∞|m) are not completely
equivalent.
16


## Page 17

where Ni1 = ∂B∗(x1,x2,...,xn|m)
∂xi
. Then the original formula is expressed as:
B(t1, t2, . . . , tn|m) ≈
1
(n −1) Pn
i=1
Ni1
ti−bi −ki
(13)
Given the minimal change in the derivative within the observable range, Ni1 is treated as a constant
Ni in this task for simplicity. Experimental results show that, if sub-RBs are separated independently,
ki is typically 0. Since ti cannot be directly quantified, we use basic form of B(ti|m) as its quantized
substitute, thus simplifying the combination law as:
B(t1, t2, . . . , tn|m) ≈
1
(n −1) Pn
i=1
Ni
B(ti|m)−bi
(14)
A.3
Calculation of RB in Practical Process
The number of planning step (𝓑(p))
0
2
4
6
8
10
12
14
2
4
6
8
10
12 14
16
The number of medical 
entity (𝓑(e))
Figure 12: Extended verification of
combination law on Medical Knowl-
edge Probing [Cheng et al., 2024]
tasks.
To determine the constants, we first fit parameters to a model
using a development dataset (or 20% of the test dataset if the
development dataset is not available). This fitting process
yields the corresponding constants. For a given task and
prompt strategy, these constants remain fixed. Additionally,
once the combination law constants are established, differ-
ent reasoning boundaries are determined through a binary
search on performance in a standard setting (3-shot CoT).
For instance, we use binary search to identify a reasoning
boundary that ensures the accuracy of all problems below
that boundary approaches 90%, achieving BAcc=90%. For
one model, one task, and one prompt type, the reasoning
boundary remains fixed. Zero-shot and few-shot settings
share the same set of reasoning boundaries.
B
The Application Tutorial of Reasoning Boundary
From a practical standpoint, our mechanism framework exhibits universal adaptability, making it
suitable for application in a wide range of scenarios. When confronted with a new problem context,
the framework enables a systematic approach to problem-solving. A key feature of the framework is
its reliance on the weighted harmonic mean, which imparts advantageous mathematical properties
to its structure. Specifically, the framework operates effectively if the reasoning process can be
segmented into relatively independent boundaries. This segmentation allows the framework to be
fully leveraged in addressing diverse problems.
Reasoning Boundary Application.
In the case of a vertical domain problem based on CoT
reasoning, the process can be divided into two key boundary levels: task planning and domain-
specific reasoning. These can be modeled as follows:
B =
1
1
Bp +
1
Bv + k1
,
(15)
where: Bp represents the task planning boundary, Bv represents the vertical domain boundary, and k1
is a constant reflecting the degree of boundary independence.
Reasoning Boundary Definition & Segmentation.
Neglecting any of these boundaries only results
in an increase in k, but keeping the overall efficiency of the framework. If the reasoning boundary is
well-defined and independent, the value of k approaches zero, showcasing the effectiveness of our
mechanism framework.
Further Reasoning Boundary Segmentation.
Further refinement of the vertical domain boundary,
Bv, into Bv1 and Bv2 is straightforward. No additional complexity is introduced, as the following
relationship holds:
Bv =
1
1
Bv1 +
1
Bv2 + k2
.
(16)
17


## Page 18

Thus, the overall boundary equation can be extended to:
B =
1
1
Bp +
1
Bv1 +
1
Bv2 + k1 + k2
.
(17)
This formulation allows for flexible and systematic boundary division at multiple levels, enhancing
the framework’s practical utility across various problem domains.
Challenging Reasoning Boundary Measurement.
In addition, we propose an alternative method
to measure the reasoning boundaries. This approach allows the model to provide direct answers
without relying on CoT reasoning steps. By doing so, the model’s reasoning process for a specific
task depends solely on a single reasoning boundary, which can be represented as follows:
B =
1
1
B1 + k1
.
(18)
For instance, in the MGSM task, assessing multilingual reasoning boundary is particularly challenging.
To address this, we directly evaluate the model’s performance using a direct prompting strategy
without CoT outputs and use this performance to define the multilingual reasoning boundary, which in
turn helps determine the corresponding normalization constant. Subsequently, we apply multilingual
CoT reasoning to the MGSM task to calculate the combined boundary using the framework’s
combination law. This approach provides a more generalized solution and may be more adaptable to
specific needs.
C
Details of Dataset
C.1
Dataset Construction
To adequately assess the reasoning boundary of LLMs, it is essential to develop a dataset that
encompasses a range of complexities and reasoning boundaries. To address these challenges, we
propose a novel approach to constructing a mathematical reasoning dataset using manual synthesis
and annotation which finally leads to the BIGGSM benchmark. Specifically, our proposed method
involves the manual synthesis and annotation of a mathematical reasoning dataset. The construction
process includes the following steps:
Step 1: Domain Template Generation
Initially, we employ a prompt-driven LLM (GPT-4) to
generate complex scenarios necessitating multi-step calculations. This process also yields initial
example templates. Specifically, the prompt given to the large model is as follows:
Generate a scenario-related template involving multiple mathematical steps to solve a real-
world problem. Ensure the scenario requires the application of different mathematical concepts.
Please use "[VAR]" as a variable to mark the template of the question.
Step 2: Natural Language Template Creation
Recognizing that LLMs can produce errors
and logical inconsistencies, we refine these initial templates to improve their accuracy and add
mathematical calculations. To facilitate the generation of extended sequences, we decompose the
templates into smaller, loopable segments that incrementally meet the multi-step reasoning demands.
Step 3: Domain Template Augmentation
To address the limited diversity in individual samples
and provide a broader evaluation of LLMs’ mathematical abilities, we use an LLM (GPT-4) to
generate at least three alternative augmented templates for each original template and step. The
generation prompt we use is as follows:
Create three alternative versions of the following template that introduce different complexities
or variables, ensuring each version demands an equivalent level of reasoning.
Step 4: Numeric Filling
Once all templates are prepared, we aim to test the upper-bound of the
LLMs’ computational reasoning boundary by introducing numerical values ranging from 1 to 1e5 in
multiplication tasks. This step is designed to thoroughly assess the models’ performance across a
spectrum of numerical challenges.
18


## Page 19

x
(b) Distribution of correct predictions
for x+y samples.
(c) Distribution of correct predictions
for x-y samples.
y
Correct sample
Incorrect sample
CFRB
PFRB
CIRB
(a) Distribution of correct predictions
for x/y samples.
200
y
x
150
100
50
0
0
0.5e5
1e5
1.5e5
2e5
x/y
=110
x/y
=500
0
0.25
0.50
0.75
1.00
1e16
0.6
0.4
0.2
0
1.0
0.8
1e16
1.0
0.5
0
2.0
1.5
1e16
1e16
0
0.2
0.4
0.6
1.00
0.8
x
Figure 13: Existence verification for reasoning boundaries on basic arithmetic calculation tasks,
including division, addition, and subtraction operations.
5
20
35
50
65
80
95
0
0.1
0.2
0.3
5
20
35
50
65
80
95
0
0.2
0.4
0.6
0.8
1
𝓑!""#$%%
𝓑!""'(%%
Accuracy (%)
Accuracy (%)
GSM8k
MATH
BigGSM
GSM8k
MATH
BigGSM
Math LLM
General LLM
Closed LLM
Open LLM
(a) Correlation between the values of CIRB 
𝓑!""#$%% for different general LLMs and 
performance on real benchmarks.
(b) Correlation between the values of CFRB 
𝓑!""'(%%for different closed and open
LLMs and performance on real benchmarks.
O1-preview
O1-preview
GPT-4o
GPT-3.5
GPT-4o
GPT-3.5
Figure 14: Correlation between the values of RB for different models and performance on real
benchmarks.
Step 5: Manual Annotation
To ensure the quality and logical coherence of our synthetic samples,
we manually review them to correct any errors introduced during the automated generation process.
Finally, we hired three experts to mark whether the samples in the data set were correct. Only for
those samples where more than two experts agreed did we retain the corresponding samples. The
Cohen’s kappa value marked by the experts was 0.97, which indicates the perfect agreement.
C.2
Dataset Analysis
Our dataset comprises 610 test samples, which is extensive when compared to the GSM8K dataset. It
features a broader range of procedural steps, varying from 1 to 16 steps. Additionally, our dataset
encompasses a wider spectrum of computational efforts, ranging from 6 to 3e5.
D
The Implementation Details of Basic Arithmetic Calculation
D.1
Data Construction
This section outlines the process used to construct datasets for examining the existence of reasoning
boundaries (RB) in basic arithmetic calculations. Initially, we identify the operations for investigation,
namely addition, subtraction, multiplication, and division. We then determine the range of integer
operands (x and y), starting from 1 to 1e10, subsequently extending to 1e20. A random number
generator is employed to produce independent and unbiased pairs of x and y within the specified
range. For each pair, we compute the expected correct outcome of the chosen operation using standard
19


## Page 20

arithmetic procedures. In addition, in order to ensure that decimals do not affect the computational
complexity, we restrict our analysis to integer operands and outcomes to control for complexity and
randomly generate numerical values of x and y.
D.2
Prompt Construction
The prompt configuration in our study involves inputting the structured data into a computational
model to analyze the arithmetic accuracy. The following prompting is used for LLMs’ input:
Please calculate the formula given below:
x op y=
where op denotes the arithmetic operation (selected from addition, subtraction, multiplication,
division). And x and y values are generated from Section D.1. The final experimental results are
shown in Figure 13.
E
The Implementation Details of Multi-hop Reasoning
We propose that the natural language multi-hop CoT task comprises two sub-tasks: multi-hop
planning and knowledge step reasoning for multi-hop question answering. To address the challenge
of measuring knowledge difficulty, we utilize a NER model4 to identify the number of knowledge
entities in each hop, thus marking the knowledge step reasoning RB in the single-step task. Formally,
let B(h) represent the RB of multi-hop planning and B(e) denote the RB of knowledge step reasoning.
The combined RB satisfies the following combination law:
BCoT(e, h) =
1
N1
(B(e)−b1) +
N2
(B(h)−b2)
.
(19)
F
Analysis for Complex-CoT and Least-to-Most within Reasoning Path Optimization
Perspective
Complex CoT Prompting can achieve better CoT within a specific RB by simplifying the
calculation reasoning step.
We believe that Complex CoT optimizes the performance of the model
by allowing the model to reach its computational limit as much as possible in single-step reasoning.
Therefore, the combined RB for Complex-CoT can be expressed as:
BComplex(p, c) =
lim
B(c)→BAcc=100%(c)
1
N1
(B(c)−b1) +
N2
(B′(p)−b2)
(20)
Assuming the premises of RB remain unchanged (BComplex(p, c) = BCoT(p, c)), it can obviously yield
the solution B′(p) > B(p). Therefore, the model can accept more steps of reasoning boundary, that
is, if the planning difficulty dp is less than reasoning capability B′(p), the accuracy is higher. In
order to analyze this problem, we adopted a meta-analysis method. We count the performance of
the work of Jin et al. [2024], Fu et al. [2023] using Complex CoT. The relationship between the
performance label and the number of steps is shown in Figure 6 (left). For most multi-step reasoning
tasks, generally speaking, within a certain range, as the number of steps increases, the computational
pressure of the model is relieved and the performance is improved, which is consistent with the theory
and exploration of Feng et al. [2024], Wang et al. [2023b], Valmeekam et al. [2023].
However, We can also clearly recognize the flaws of Complex CoT. Once the difficulty of planning
dp (that is, the number of planning steps) is greater than B′(p), it exceeds the capabilities of the
model and the performance will decline. We can observe that for single-step calculation reasoning,
as shown in Figure 6 (right), the performance of using Complex CoT will gradually decrease. The
rest of mathematical reasoning will also decrease when the number of steps is greater than a certain
threshold. This phenomenon can also be explained by our combination law. While the amount of
calculation is reduced, the number of reasoning steps is also increasing. If the acceptable number of
reasoning steps is exceeded, the reasoning boundary is exceeded, and the model performance will
4https://huggingface.co/dslim/bert-base-NER
20


## Page 21

64%
31%
3%
1%
1%
(a) Distribution of the number of CoT
steps performed for each sub-question.
#3
#5
#≥7
#6
#4
64.32 
61.9
36.84 
20.00 
0.00 
0
10
20
30
40
50
60
70
(b) The Accuracy Distribution on the CoT steps
performed for each sub-question.
2%
29%
17%
12%
8%
14%
9%
9%
(c) Distribution of the number of
sub-question.
#≤3
#8
#5
#7
#6
#4
#9
#≥10
#≤3
#5
#≥7
#6
#4
Accuracy (%)
Figure 15: Analysis of output results for Least-to-Most Prompting.
decline, which demonstrates that it is necessary to keep a balance between the number of reasoning
steps and computational pressure (see Appendix G for detailed meta-analysis process).
Limitation:
Need to keep balance in the number of reasoning steps and calculation pressure.
Least-to-Most Prompting can achieve better CoT within a specific RB by simplifying the
planning reasoning paths.
Least-to-most prompting structures problem-solving hierarchically, by
breaking questions into smaller sub-questions and further solving them one-by-one. Accordingly, the
Least-to-most RB can be divided into three sub-RBs, namely, the problem decomposition RB B(d),
the problem planning RB B(p), and the single-step calculation RB B(c). Therefore, the combined
RB for least-to-most can be expressed as:
BLtM(d, p, c) =
1
2N1
(B′(c)−b1) +
2N2
(B(p)−b2) +
2N3
(B(d)−b3)
.
(21)
Ideally, if the problem decomposition ability of the model is excellent (B(d) →+∞), it can
decompose the problem into sub-problems that can be solved in one step every time B(c) →1,
therefore the least-to-most RB can be expressed as:
ˆBLtM(d, p, c) =
lim
B(c)→1,B(d)→+∞BLtM(d, p, c) =
B′(c) −b2
N1(B′(c) −b2) −N2
,
(22)
Assuming the premises of RB remain unchanged ( ˆBLtM(d, p, c) = BCoT(p, c)), it can obviously yield
the solution B′(c) > B(c). On the contrary, the model can accept larger difficulty d, which also shows
that using least-to-most prompting can effectively increase the maximum of acceptable calculation
RB under a given RB (as shown in Figure 7), thereby improving model performance. As shown in
Table 1, we find that LLM can be optimized by Least-to-most from vanilla CoT.
However, the performance improvement of the model is not significant, which we attribute to the
fact that the current model cannot push its performance to the ideal limit. As shown in Figure 15 (a),
the reasoning boundary of the model cannot make each reasoning step completely tend to 1, which
also leads to the difference in reasoning performance in Figure 15 (b). In the meantime, the model’s
ability to divide problems is also limited. What’s more, as shown in Figure 15 (c), in around 90% of
cases, the model will only divide less than 6 problems, which also limits the performance.
Limitation:
Although the pressure of local planning has been reduced, it has not actually
effectively reduced the pressure of global planning, nor the pressure of optimization calculations.
G
The Meta-Analysis for Complex-CoT Prompting
In order to discuss the advantages and limitations of Complex-CoT, we conduct two detailed meta-
analyses through two distinct perspectives. The first one assesses the influence of the reasoning
steps demonstrated in In-Context Learning (ICL) across various tasks, while the second evaluates the
effects of employing a fixed number of reasoning steps in ICL on questions with different reasoning
steps. These meta-analyses aim to compare the efficacy of these methods against prior studies
21


## Page 22

text-davince-002
text-davince-003
GPT3.5
Manual Complex CoT
Auto-Generated Complex CoT
GPT4
The number of steps
The number of steps
10
30
50
70
90
Accuracy (%)
0
1
2
3
4
5
6
90
91
92
93
94
95
The number of steps
Accuracy (%)
0
1
2
3
4
5
6
90
92
94
96
98
Accuracy (%)
0
1
2
3
4
5
6
(c) MultiArith
(a) GSM8K
(b) SingleEq
(f) AQuA
(d) SAVMP
50
55
60
65
70
75
The number of steps
Accuracy (%)
0
1
2
3
4
5
6
81
82
83
84
85
86
The number of steps
Accuracy (%)
0
1
2
3
4
5
6
65
66
67
68
69
70
71
The number of steps
Accuracy (%)
0
1
2
3
4
5
6
(e) StrategyQA
Figure 16: The effectiveness on average step length in demonstrations for CoT performance.
systematically. Specifically, we conducted a systematic search for relevant studies addressing the
same problem tackled by Jin et al. [2024], Fu et al. [2023], Shum et al. [2023], Sun et al. [2023],
and Jiang et al. [2023b]. We ensured that retrieved studies were pertinent, focusing on studies that
addressed the same problem and used similar evaluation metrics.
G.1
The effectiveness of step length in demonstrations
From each selected study, including Jin et al. [2024] and Fu et al. [2023], we evaluate the performance
using Complex CoT. The relationship between performance and the number of Complex CoT’s steps
is shown in Figure 16. For most multi-step reasoning tasks, as the number of steps increases within a
certain range, the computational load decreases, and performance improves.
However, as described in Appendix F, the flaws of Complex CoT are apparent. When the difficulty
of planning (dp), defined as the number of planning steps, exceeds B′(p), the model’s capabilities are
surpassed, leading to a performance decline. This is evident in single-step calculation reasoning, as
shown in Figure 16 (c, e), where performance using Complex CoT gradually decreases. Similarly,
for other mathematical reasoning tasks, performance decreases when the number of steps exceeds a
certain threshold. This phenomenon aligns with our combination law: while reducing the amount
of calculation, the number of reasoning steps increases. Exceeding the acceptable number of
reasoning steps surpasses the reasoning boundary, causing a decline in model performance. Therefore,
maintaining a balance between the number of reasoning steps and computational pressure is crucial.
G.2
The effectiveness on step length in golden samples
Furthermore, to gain a nuanced understanding of the impact of Complex CoT when the number
of steps exceeds the golden step number, we conduct further meta-analysis from Fu et al. [2023],
Shum et al. [2023], Sun et al. [2023], Jiang et al. [2023b]. Specifically, as illustrated in Figure 17
(a, b), our analysis reveals that for problems of low complexity and with smaller golden step
numbers, Complex CoT tends to underperform compared to Vanilla CoT. Notably, it is only when the
reasoning steps exceed two that Complex CoT outperforms Vanilla CoT. This suggests that Complex
CoT effectively optimizes single-step computations and enhances model performance for complex
22


## Page 23

20
30
40
50
60
2
3
4
5
6
7
8
10
30
50
70
90
LLaMA2
text-davince-002
text-davince-003
Vanilla CoT
Manual Complex CoT
GPT3.5
The number of steps
Accuracy (%)
The number of steps
Accuracy (%)
2
3
4
5
(a) GSM8K
(c) AQuA
2
3
4
5
6
7
8
10
30
50
70
90
The number of steps
Accuracy (%)
(b) GSM8K
LLaMA
Figure 17: The effectiveness of step length in golden samples with a fixed step length in demonstra-
tions for CoT performance.
[Minimum Reasoning Path Prompting]
You need to perform multi-step reasoning, with each step carrying out as many basic operations as 
possible.
[Acceptable Reasoning Prompting]
Remember, you can only complete tasks that contain up to 5 basic operations per step, and 
multiplication operations must be less than 1.5e5. The upper limit of the multiplication operations 
decreases as the number of operations per step increases.
[EXAMPLE]
Question: Leo's assignment was divided into three parts. He finished the first part of his 
assignment in 25 minutes. It took him twice as long to finish the second part. If he was able to 
finish his assignment in 2 hours, how many minutes did Leo finish the third part of the assignment?
Answer: Leo finished the first and second parts of the assignment in 25 + 25*2 = 
<<25+25*2=75>>75 minutes.
Therefore, it took Leo 60 x 2 - 75 = <<60*2-75=45>>45 minutes to finish the third part of the 
assignment.
#### 45
…
[REQUEST]
Question: [Question]
Figure 18: Minimum acceptable reasoning path prompting for natural language chain-of-thought. All
examples given in the context transform from Wei et al. [2022].
problems. However, it increases the cognitive load for simple problems, resulting in a performance
decline.
Interestingly, this phenomenon is also observed with the simpler Vanilla CoT, as shown in Figure 17
(c). The model achieves significant performance gains only when the number of reasoning steps
aligns with the target output steps. If the complexity of the planned steps exceeds the necessary
reasoning boundary, or if there is no effective optimization for reasoning boundary, the performance
deteriorates.
G.3
The Implementation Details of Minimum Acceptable Reasoning Paths
To address the two aforementioned limitations, we propose Minimum Acceptable Reasoning Paths
(MARP). Firstly, to reduce the model’s computational load, we introduce instructions that limit its
single-step computing power, thereby optimizing its reasoning boundary. Secondly, to enhance the
23


## Page 24

Model
Acc. (↑)
Input Token (↓)
Output Token (↓)
HotpotQA [Yang et al., 2018]
CoT
289.50
67.27
26.50
CoT+MARP
309.51
68.39
28.73
Medical Probing [Cheng et al., 2024]
CoT
636.11
249.78
48.9
CoT-MRP
476.11
86.52
69.41
StrategyQA [Geva et al., 2021]
CoT
1046.28
225.35
63.90
CoT+MARP
649.28
167.40
74.09
Table 2: Extended experimental results on GPT-3.5-Turbo.
model’s acceptability, we increase the computation amount per step within this boundary and reduce
the number of global planning steps, thus alleviating planning pressure.
To control variables effectively, we make only the simplest modifications to the prompt to achieve the
desired CoT optimization.
Minimum Reasoning Path Prompting
To alleviate the cognitive load associated with planning,
it is essential to have the model respond to the question as succinctly as possible. This approach
ensures that the focus remains on providing a short, clear and direct reasoning path. The following
prompt is designed to achieve this objective:
You need to perform multi-step reasoning, with each step carrying out as many basic operations
as possible.
Acceptable Reasoning Prompting
To effectively utilize the model, it is crucial to define the
upper-bound of reasoning boundary. This ensures that the complexity of the reasoning process is
manageable and within acceptable bounds. The specific prompt to achieve this is as follows:
Remember, you can only complete tasks that contain up to 5 basic operations per step, and
multiplication operations must be less than 1.5e5. The upper limit of the multiplication
operations decreases as the number of operations per step increases.
This prompt is designed to set clear boundaries for the model’s operations, thereby optimizing its
performance and accuracy.
Furthermore, it is necessary to enhance the demonstration within the corresponding in-context
learning framework to meet the specific needs of our Model-Agnostic Reasoning Protocol (MARP).
This involves refining the examples and instructions provided to ensure they align perfectly with the
MARP requirements. Figures 18 and Figure 19 illustrate our MARP prompt, showcasing how to
structure the demonstrations to facilitate effective learning and reasoning in natural language CoT and
program-of-thought setting. By adhering to these guidelines, we can ensure that the model operates
efficiently and produces reliable results.
In summary, setting precise boundaries for reasoning boundary and optimizing in-context learning
demonstrations are essential steps in enhancing the model’s performance. By following the specified
prompt and refining the MARP examples, we can achieve a high level of accuracy and efficiency in
the model’s reasoning processes.
H
The Implementation Details in various LLMs
We employ 25 commonly used models to evaluate the extensibility of our framework to a broader
range of models. The specific models are listed in Table 3. For each model, we utilize the chat/instruct
version whenever available to maximize their ability to follow instructions. Additionally, we deploy
all models on the vLLM [Kwon et al., 2023] framework to ensure a fair comparison. Except for
model OpenMath-series [Toshniwal et al., 2024] which does not conform to the vLLM format, all
other models are deployed on vLLM for testing. All experiments on open-source models were
24


## Page 25

[Minimum Reasoning Path Prompting]
You need to perform multi-step reasoning, with each step carrying out as many basic operations as 
possible.
[Acceptable Reasoning Prompting]
Remember, you can only complete tasks that contain up to 5 basic operations per step, and 
multiplication operations must be less than 1.5e5. The upper limit of the multiplication operations 
decreases as the number of operations per step increases.
[EXAMPLE]
Question: Leo's assignment was divided into three parts. He finished the first part of his 
assignment in 25 minutes. It took him twice as long to finish the second part. If he was able to 
finish his assignment in 2 hours, how many minutes did Leo finish the third part of the assignment?
Answer: Leo finished the first and second parts of the assignment in 25 + 25*2 = 
<<25+25*2=75>>75 minutes.
Therefore, it took Leo 60 x 2 - 75 = <<60*2-75=45>>45 minutes to finish the third part of the 
assignment.
#### 45
…
[REQUEST]
Question: [Question]
Figure 19: Minimum acceptable reasoning path prompting for program-of-thought. All examples
given in the context transform from Wei et al. [2022].
conducted on two A100 80G. Following the setting of Wei et al. [2022], in our CoT experiment, all
multi-step reasoning tasks are with three manually constructed demonstrations. In addition, for all
the experiments, our top-p is selected from {0.95, 1}, and temperature is selected from [0, 1].
In addition, the only difference in the prompt is that we use different dialogue delimiters to make it
conform to the format of the LLM instruction fine-tuning, thereby avoiding the bias caused by the
gap between training and inference.
Model
Base Model
Parameters (B)
Open-source General LLM
LLaMA [Touvron et al., 2023a]
-
7, 13, 33, 65
LLaMA-2 [Touvron et al., 2023b]
-
7, 13, 70
LLaMA-3 [Meta, 2024]
-
8, 70
Code-LLaMA [Roziere et al., 2023]
LLaMA-2 [Touvron et al., 2023b]
7, 13, 34, 70
Mistral [Jiang et al., 2023a]
-
7
Close-source General LLM
Gemini-1.0-Pro [Team et al., 2023]
-
-
GPT3.5-Turbo [OpenAI, 2022]
-
-
Claude-3-Haiku [Anthropic, 2024]
-
-
Claude-3-Sonnet [Anthropic, 2024]
-
-
Claude-3-Opus [Anthropic, 2024]
-
-
GPT4 [OpenAI, 2023]
-
-
Open-source Math LLM
MAmmoTH [Yue et al., 2023]
LLaMA-2 [Touvron et al., 2023b]
7,13
MAmmoTH [Yue et al., 2023]
Mistral [Jiang et al., 2023a]
7
OpenMATH-Instruct [Toshniwal et al., 2024]
LLaMA-2 [Touvron et al., 2023b]
70
OpenMATH-Instruct [Toshniwal et al., 2024]
Mistral [Jiang et al., 2023a]
7
Table 3: Model list. In order to ensure a certain ability to follow instructions, we use the Instruct
version of the model as much as possible (if available).
25


## Page 26

I
The Implementation of Combination Law in MGSM
Inspired by Qin et al. [2023] and Huang et al. [2023], we propose that the multilingual mathematical
CoT task comprises three sub-tasks: step planning, step calculation, and multi-modal expression. We
evaluate the model’s mathematical expression ability in different languages based on its zero-shot
direct performance on MGSM, as reported by Qin et al. [2023]. For relevant parameter calculations,
please see the “Challenging Reasoning Boundary Measurement” part of Appendix B. Formally, let
step planning RB be denoted by B(p), step calculation RB by B(c), and multilingual expression RB
by B(l). The combined RB satisfies the following law:
BCoT(c, p, l) =
1
2N1
(B(c)−b1) +
2N2
(B(p)−b2) +
2N3
(B(l)−b3)
.
(23)
As shown in Figure 10, the performance distribution of RB (including BAcc=90% and BAcc=10%) in the
multilingual mathematical reasoning task aligns with the proposed combination law in Equation (23).
Moreover, three distinct RBs are evident in Figure 10.
J
Ethical Considerations
Data Access.
Our data is adapted from GSM8K [Cobbe et al., 2021] and supplemented with
manually created samples. GSM8K is an open-source dataset available for academic research.
Dataset Collection Process.
We began with an introductory task interview using 50 example
questions, compensating participants $20 each to familiarize themselves with the task. During the
annotation process, annotators were paid $15 per hour, totaling approximately 60 hours of work.
The Rest of Data Annotation Process.
For the remaining data annotation, we hired a graduate
student with CET-6 proficiency in Chinese and English and strong mathematical knowledge. The
student was compensated $15 per hour, which is above the local average salary. The instructions for
annotation are as follows:
You need to annotate the generated number of steps, maximum computation amount, correctness
of the generation steps, correctness of the calculations, and correctness of the model output:
• Number of generated steps: This refers to how many reasoning steps the model generated.
• Maximum computation amount: This indicates the largest product of operations in the
model’s reasoning steps.
• Correctness of generation steps: This assesses the accuracy of the model’s planning. If
all steps and operators are planned correctly, and the operand values are logically correct,
it is considered correct, regardless of calculation accuracy.
• Correctness of calculations: This considers only whether the calculations are correct,
ignoring planning factors.
• Correctness of the output: This checks whether the model’s final answer is correct.
26


## Page 27

NeurIPS Paper Checklist
1. Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: As shown in lines 5-18 of the Abstract and 59-69 of the Introduction, we
present our main claims and outline the paper’s contributions and scope.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2. Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: We have discussed the limitations of our work in Section 8.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.
3. Theory Assumptions and Proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [NA]
27


## Page 28

Justification: Our work is not strictly a purely theoretical work, we provide more of an
empirical formula. In addition, we analyze the source of our empirical formula in Appendix F
and provide the corresponding assumption and proof.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4. Experimental Result Reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: As shown in Appendix C to Appendix I, we have provided detailed descriptions
and analyses of the experimental setups for all our investigations.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5. Open access to data and code
28


## Page 29

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [No]
Justification: We will release our code in the official version of the subsequent paper to
provide reproduction and provide more help to the future community.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6. Experimental Setting/Details
Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]
Justification: As shown in Section C to Section H, we have provided detailed descriptions
and analyses of the experimental setups for all our investigations.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [Yes]
Justification: We report error bars in Table 1 and Figure 4, and explain the error variables
in Section 3. However, error bars are not reported for all tasks because it would be too
expensive for human annotation and computational resource consumption.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
29


## Page 30

• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8. Experiments Compute Resources
Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: As shown in Section 3 and Appendix H, we provide detailed model compute
resources under different settings.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9. Code Of Ethics
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: We are convinced that we comply with NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader Impacts
Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes]
Justification: We have discussed the broader impacts of our work in Section 8. In addition,
since our work is more like providing an empirical formula and has no additional social
harmfulness, we do not discuss this part.
Guidelines:
30


## Page 31

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11. Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12. Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [NA]
Justification: The paper does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
31


## Page 32

• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and Research with Human Subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [Yes]
Justification: As shown in Section J, We describe and analyze the details and ethical
considerations of our crowdsourcing in detail.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [No]
Justification: Since our region and institution are not required to provide IRB approval, we
do not describe this section. We are convinced that our work complies with the NeurIPS
Code of Ethics and the guidelines.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
32


## Page 33

• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.
33



---

# Paper 2: 2510.04028v1(debate on RLVR)


**Source file:** `2510.04028v1(debate on RLVR).pdf`

**Pages:** 28


## Page 1

The Debate on RLVR Reasoning Capability Boundary:
Shrinkage, Expansion, or Both? A Two-Stage Dynamic View
Xinhao Yao1,2*
Lu Yu2
Xiaolin Hu3
Fengwei Teng1
Qing Cui2
Jun Zhou2
Yong Liu1†
1Renmin University of China
2Ant Group
3Xiamen University
Abstract
The ongoing debate on whether reinforcement learning with verifiable rewards (RLVR) expands or shrinks
the reasoning capabilities of large language models (LLMs) remains unresolved. Some studies contend that
RLVR mainly improves sampling efficiency but at the expense of diversity and exploratory capacity, resulting in
capability boundary shrinkage. In contrast, others demonstrate that prolonged training can lead to the emergence
of novel reasoning strategies, suggesting capability boundary expansion. To reconcile these contradictory
findings, we theoretically and empirically show that both perspectives are partially valid—each aligning with
a separate phase in an inherent two-stage probability mass dynamic: (1) Exploitation stage: initially, the
model primarily samples explored high-reward and low-reward tokens, while rarely selecting the potentially
optimal token. Positive advantage estimates increase the probability of high-reward tokens and decrease those
of low-reward tokens, yet the optimal token’s probability remains largely unchanged during this stage. (2)
Exploration stage: as training advances, the growth rate of previously acquired high-reward tokens slows as
their probabilities approach saturation. When a potentially optimal token—now receiving positive advantage
estimates—is occasionally sampled, its probability increases, while those of the originally high-reward tokens
decrease. This dynamic suggests that over-exploitation during the exploitation stage may lead to capability
boundary shrinkage, whereas prolonged training into the exploration stage can promote an expansion of the
reasoning capability boundary. Building upon our insights, we revisit the potential of only using relative negative
gradients for prolonging training, providing a theoretical and empirical foundation for the development of more
advanced reasoning capabilities.
1
Introduction
Reinforcement learning with verifiable rewards (RLVR) has become a key paradigm for substantially enhancing the
reasoning abilities of large language models (LLMs), as exemplified by advanced models such as OpenAI’s O1 and O3
[32, 44] and DeepSeek-R1 [21]. By optimizing pre-trained or chain-of-thought (CoT) [63] fine-tuned models through
verifiable reward signals, RLVR enables LLMs to excel in complex logical tasks such as mathematics [40, 79, 80] and
programming [35, 39].
Despite empirical successes, a fundamental question is still hotly debated: does RLVR genuinely expand the reasoning
capabilities of base models beyond their original boundaries? Current evidence is sharply divided. (1) One line of
research [78, 83, 13, 24, 42, 53, 20] argues for capability boundary shrinkage, contending that while RLVR improves
sampling efficiency, it fails to produce genuinely novel reasoning strategies and may even induce a progressive narrowing
of reasoning capabilities during training. Empirical evidence from Yue et al. [78] shows that although RLVR-trained
models perform better under small-k sampling (e.g., k = 1), base models achieve higher Pass@k when k is large.
Similarly, Cui et al. [12] document a sharp entropy collapse during training, resulting in overly deterministic behavior
[81] and reduced exploratory effectiveness. (2) In contrast, another body of work [37, 64, 36, 66, 77, 61, 57] provides
evidence supporting capability boundary expansion. Liu et al. [37] attribute previous evidence of capability boundary
shrinkage to the premature termination of RL training, which disrupts learning before novel reasoning capabilities can
fully develop. Through prolonged training, they further demonstrate that RLVR can explore and populate new regions of
*Work done during an internship at Ant Group.
†Corresponding author: liuyonggsai@ruc.edu.cn.
1
arXiv:2510.04028v1  [cs.LG]  5 Oct 2025


## Page 2

A preprint
solution space over time. Meanwhile, Wu et al. [66] experimentally show that RLVR can occasionally expand empirical
support, producing novel correct solutions beyond the original reach of the base model.
The debate between these two lines of evidence centers on empirical results; however, the underlying mechanisms
responsible for these contradictory findings remain unclear. To elucidate the mechanisms, we focus on the evolution
of the policy model’s probability mass distribution—termed the probability mass dynamics. As a conceptual starting
point, consider that the search tree [78, 87, 23] for any given prompt is built through iterative sampling from the policy.
This tree grows exponentially at a rate of O(V T ), where V denotes the vocabulary space (token set) size and T the
maximum generation length. Crucially, policy updates can be viewed as a dynamic reallocation of probability mass
across the search tree, thereby shaping the reasoning capability boundary.
Through an integrated theoretical and empirical analysis (Section 3), we demonstrate that both lines of evidence
hold validity to some extent—each corresponding to a distinct stage within a two-stage dynamic of probability mass.
Specifically, since the logit for token v is directly tied to its policy probability—a larger (smaller) logit results in a
higher (lower) probability—we analyze the policy gradient of the training objective and derive a bidirectional update
rule for the logits (i.e., the pre-Softmax values; Lemma 1). According to this rule, updates to the logits depend on both
the advantage estimate ˆA and the current policy distribution π. Under practical optimization settings such as GRPO
[54] (where multiple responses are sampled per prompt), Theorem 1 establishes that the expected logit update for token
v is proportional to π(v)
h
(1 −π(v)) ˆA(v) −P
u̸=v π(u) ˆA(u)
i
.
From this view, the overall dynamic appears to unfold in two distinct stages. (1) Exploitation stage: initially, the model
predominantly samples the already-explored high-reward token and the low-reward token, while the potentially optimal
token is selected only infrequently. Driven by positive advantage estimates, the probability of the high-reward token
increases, whereas that of the low-reward token decreases. However, the probability of the potentially optimal token
remains largely unchanged throughout this stage. This behavior suggests that over-exploitation during this stage may
result in a shrinkage of the capability boundary. (2) Exploration stage: as training progresses, the growth rate of the
high-reward token previously explored slows as its probability approaches near saturation (1 −π →0). When the
potentially optimal token—now associated with positive advantage estimates—is occasionally sampled, its probability
increases, while that of the formerly high-reward token declines. A key characteristic of this dynamic is the transition
of the relative negative sample: from the initially low-reward token to the high-reward token. This implies that with
prolonged training, gradient updates can be progressively redirected toward tokens with low initial probability but high
potential, once high-probability tokens have stabilized, ultimately expanding the reasoning capability boundary. We
illustrate these theoretical insights with a toy example (Section 3.2).
Building on our theoretical and experimental insights, a direct way to expand the reasoning capability boundary and
mitigate shrinkage is to prolong training while concentrating policy probability updates exclusively on optimizing
relative negative samples (denoted -N, Section 4.1) throughout the learning process. Empirical investigations (Section
4) of our strategy—implemented in widely adopted algorithms (e.g. GRPO, GSPO [86]) on benchmark datasets and
open-source LLMs verify that GRPO-N (GSPO-N) achieves competitive and stable performance improvements while
largely preserving the base model’s diversity, demonstrating the potential for prolonged training. Notably, analysis of
the training process reveals instances where incorrect code is initially generated but is later refined and corrected through
iterative reflection. Unlike GRPO, which reinforces the entire trajectory—including error-prone steps—GRPO-N
effectively prevents such reinforcement.
⋄Main contributions. Briefly, this study unveils the underlying mechanisms responsible for the heated debate (boundary
shrinkage or expansion) in RLVR from both theoretical and practical perspectives. We emphasize the essential role of
fine-grained probability mass allocation and establish a theoretical and empirical basis for understanding the impact of
RLVR on reasoning capabilities.
1.1
More Related Works
Broadly speaking, our work builds upon lines of research in reinforcement learning for LLM reasoning, LLM learning
dynamics, and gradient analysis in preference optimization. A comprehensive review of related work is included in
Appendix A due to page constraints.
2


## Page 3

A preprint
2
Preliminaries and Background
In this section, we describe the core components of our study by reviewing some basic notations.
RLVR. Reinforcement learning with verifiable rewards (RLVR) is a paradigm for improving models on tasks with
objectively verifiable outcomes. In this formulation, an autoregressive language model is treated as a policy πθ
(parameter θ). For a given query x from a prompt set D, the probability of generating a response y is defined as
πθ(y | x) = Q|y|
t=1 πθ(yt | x, y<t). A deterministic reward function r assigns a scalar value indicating the correctness
of the full response y to the prompt x. Each token in y receives the same reward (1 only if the final answer is correct, and
0 otherwise). The objective is to minimize the loss: LRLVR(θ) = −Ex∼D, y∼πθ(·|x) [r(x, y)], where r(x, y) ∈[0, 1].
A unified framework for policy gradient optimization. Building on the work of [54, 34, 58], we consider a unified
objective J that establishes connections among various optimization methods:
JRLVR(θ) = Ex∼D, y∼πθold(·|x)

1
|y|
|y|
X
t=1
min

wt(θ) ˆAt, clip
 wt(θ), 1 −ϵ, 1 + ϵ
 ˆAt


,
(1)
where ϵ is a clipping hyperparameter, clip(·) is the clipping operation, and the the importance ratio of the token yt is
defined as wt(θ) =
πθ(yt|x,y<t)
πθold(yt|x,y<t) (the current policy πθ and the old policy πθold). ˆAt is the advantage of current token
and is implemented differently across optimization methods:
• PPO (Proximal Policy Optimization [52, 46]). ˆAt is computed by applying Generalized Advantage Estimation (GAE)
[51], based on the value model. This incurs considerable computational and memory overhead, and its effectiveness
critically depends on the reliability of its value estimation.
• GRPO (Group Relative Policy Optimization [54]). To reduce variance, GRPO and its variants (e.g., DAPO [76] &
Dr.GRPO [38]) eliminate reliance on a value model by using Monte Carlo estimates to compute the relative advantage
across a group of responses {yi}G
i=1 ∼πθold to the same query (where G is the group size and all token in yi share the
same relative advantage):
wi,t(θ) = πθ(yi,t | x, yi,<t)
πθold(yi,t | x, yi,<t),
ˆAi,t = ˆAi = r(x, yi) −mean
 {r(x, yi)}G
i=1

std
 {r(x, yi)}G
i=1

.
• GSPO (Group Sequence Policy Optimization [86]). Given that the token-level importance ratio wi,t in GRPO does
not align with sequence-level rewards, GSPO introduces a sequence-level importance ratio wi based on sequence
likelihood [85]:
wi(θ) =
 πθ(yi | x)
πθold(yi | x)

1
|yi|
= exp

1
|yi|
|yi|
X
t=1
log πθ(yi,t | x, yi,<t)
πθold(yi,t | x, yi,<t)

.
To better understand the model’s learning dynamics under this binary outcome reward setting, we omit the regularization
components1 (e.g., KL term & clipping operation). That is, the policy gradient ∇θJRLVR(θ) can be simplified to
E
h
1
|y|
P|y|
t=1 wt(θ) ˆAt∇θ log πθ(yt | x, y<t)
i
with respect to θ. Specifically, taking GRPO as an example (Appendix
B.1 for derivation):
∇θJGRPO(θ) = Ex,{yi}G
i=1

1
G
G
X
i=1
1
|yi|
|yi|
X
t=1
wi,t(θ) ˆAi,t
|
{z
}
coefficient
∇θ log πθ(yi,t | x, yi,<t)

.
(2)
Remark 1. Intuitively, if we set ˆAi,t = 1 and wi,t = 1 while all yi are correct responses, then Eq.(1) essentially
performs maximum likelihood estimation, i.e., supervised fine-tuning (SFT). Furthermore, Eq.(2) indicates that the
scalar wi,t ˆAi,t can be interpreted as a weighting coefficient that adjusts the log-likelihood term. This implies that
RLVR methods can be viewed as a form of reweighted SFT, where correct responses and incorrect responses contribute
positive and negative gradients, respectively [54, 10, 88, 15, 1, 8]. When ˆAi,t is calculated from a comparison of
average rewards across groups (e.g., GRPO), the resulting gradient is named the relative policy gradient.
1Regularization components are widely regarded as mechanisms for ensuring training stability. Moreover, studies [30, 10]
indicate that omitting them does not impair performance when others are properly tuned.
3


## Page 4

A preprint
3
Probability Mass Dynamics
As described above, we begin by considering a standard task that involves generating a reasoning sequence. In this
setting, the model learns a policy πθ(y | x) = QT
t=1 πθ(yt | x, y<t) ∈RV ×T to map an input x to a sequence of
predictions y = {y1, . . . , yT }, where y ∈VT , V is the vocabulary space of size V , and T denotes the maximum
generation length. Conceptually, the reasoning process can be regarded as a tree search [78, 87, 23]. A search tree
is constructed for a given problem by iteratively sampling from the policy model. This process leads to exponential
growth in the tree size, O(V T ), reflecting an open-ended and combinatorially infinite reasoning space [45].
Crucially, policy updates can be viewed as dynamically reallocating probability mass over the search tree, thereby
shaping the boundary of reasoning capability. Here, we specifically focus on the evolution of the policy model’s
probability distribution—referred to as probability mass dynamics.
⋄Learning dynamics offer critical insights into the key challenges and counterintuitive behaviors of deep learning [50],
with early explanations pointing to network “stiffness" [19] or “local elasticity" [25, 16]. To track the evolution of the
probability distribution, we monitor the logits zθ ∈RV ×T and the log probabilities log πθ(y | x), where πθ is derived
from zθ via a column-wise Softmax πθ(· | x, y<t) = Softmax(zθ(x, y<t)). The probability mass dynamics are then
defined as:
∆zl(x) ≜zθl+1(x) −zθl(x),
(3)
∆log πl(y | x) ≜log πθl+1(y | x) −log πθl(y | x),
(4)
where the model’s parameter θ is updated from step l to l + 1 by performing one policy gradient update on the sample
data (x, y). For simplicity, we primarily analyze the case where T = 1 (i.e., y ∈V), meaning ∆zl ∈RV ×1 and its
dimension aligns with the size of the model’s vocabulary. Notably, a larger (smaller) logit results in a higher (lower)
probability. For T > 1, the updates can be computed separately; therefore, we can calculate the the distinct T updates
and stack them together.
3.1
A Two-Stage Dynamic: Exploitation and Exploration
Given the monotonicity of the Softmax function, the main text focuses mainly on characterizing the changes of z.
Analysis of the updates to log πθ with respect to θ is provided in the Appendix B.3.
Lemma 1 (Logits Update for Softmax Parameterization). Consider a policy parameterized by a Softmax function over
logits z(x) := z = [z1, · · · , zV ]T , such that the probability of action (or token) v is given by π(v) := π(v | x) =
Softmax(z)v = exp (zv)/ PV
v′ exp (zv′). Reviewing Eq.(2), if the currently sampled action is v, let the policy gradient
estimate be ∇zJ ≈ˆA(v)∇z log π(v). For a learning rate η, the update rule for the logits at time step l is (Appendix
B.2 for derivation):
• For the sampled action v:
zl+1
v
←zl
v + η · ˆA(v) ·
 1 −πl(v)

,
∆zl
v = η · ˆA(v) ·
 1 −πl(v)

,
• For all other actions u ̸= v:
zl+1
u
←zl
u + η · ˆA(v) ·
 −πl(u)

,
∆zl
u = η · ˆA(v) ·
 −πl(u)

.
Remark 2 (Bidirectional Update Rule). The update to the logit z ∈RV ×1 depends on both the advantage estimate
ˆA and the current policy distribution π. Specifically, Let v denote the currently sampled action. (1) when ˆA(v) > 0:
zv increases by η ˆA(v)(1 −π(v)) while zu (u ̸= v) decreases by η ˆA(v)π(u); (2) when ˆA(v) < 0: zv decreases by
η| ˆA(v)|(1 −π(v)) while zu (u ̸= v) increases by η| ˆA(v)|π(u). The normalization property of Softmax ensures that
when ˆA(v) > 0, the update increases π(v) while decreasing π(u) for all u ̸= v, including other advantageous actions.
In contrast, when ˆA(v) < 0, the update increases the probabilities of other actions proportionally to their current
policy values. The update may reallocate probability mass toward other potentially advantageous actions that were
previously under-sampled.
The practical update in group policy optimization (e.g. GRPO, DAPO, GSPO, REINFORCE++ [29], GPG [10], GPO
[75]), which employs Monte Carlo sampling, arises from the collective effect of a group of responses, thus motivating
our analysis of the expected logits update.
4


## Page 5

A preprint
Theorem 1 (The Expected Logits Update). Under the conditions stated in Lemma 1, we assume2 that x ∼D is
i.i.d. and {ui}G
i=1 are randomly sampled from π(· | x), the expected group relative policy gradient ∇zJ ∈RV ×1 is
Ex∼D,{ui}G
i=1∼π(·|x)
h
1
G
PG
i=1 ˆA(ui)∇z log π(ui)
i
. Then the expected logits update is (proof in Appendix B.4):
E(∆zl
v) = η · πl(v)

(1 −πl(v)) ˆA(v) −
X
u̸=v
πl(u) ˆA(u)

.
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.85
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
a1 : r1 = 0.85
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
a1 : r1 = 0.85
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
a1 : r1 = 0.85
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.9
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
a1 : r1 = 0.8
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
a1 : r1 = 0.7
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
a1 : r1 = 0.6
a2 : r2 = 1.0
a3 : r3 = 0
Figure 1: The probability mass dynamics of policy optimization across varying action rewards r and initial policy
probabilities π. Each sub-figure corresponds only to the indicated rewards and probabilities. The first row compares the
impact of different initial policy probabilities under identical rewards, while the second row compares the effect of
varying rewards given the same initial policy.
Remark 3 (A Two-Stage Dynamic of Exploitation and Exploration). Theorem 1 establishes that the magnitude of
the expected logit update ∆zl
v is explicitly governed by πl(v). Although the Softmax function guarantees strictly
positive probabilities for all actions, a significant number of these actions lie within the extreme tail of the probability
distribution. As a result, under finite-sample training conditions, such actions exert negligible influence on parameter
updates and are effectively omitted during optimization (that is, πl(v) →0 leads to ∆zl
v →0). Interestingly, the
overall dynamic appears to unfold in two distinct stages. (1) Exploitation stage, corresponding to capability boundary
shrinkage: initially, the model mainly samples already-explored high-reward and low-reward tokens, rarely selecting
the potentially optimal one. Driven by positive advantage estimates, the probability of the high-reward token increases
while the low-reward token decreases. The potentially optimal token’s probability remains largely unchanged (or may
even decrease, Remark 2), suggesting that over-exploitation in this stage may cause capability boundary shrinkage.
(2) Exploration stage, corresponding to capability boundary expansion: As training continues, the growth of the
previously dominant high-reward token slows as it approaches saturation (1 −π →0). When the potentially optimal
token—now receiving positive advantage signals—is occasionally sampled, its probability rises, while that of the former
high-reward token decreases. A key feature of this stage is the shift in relative negative sampling: from the initial
low-reward token to the once high-reward token. This implies that through prolonged training, gradient updates can be
shifted toward tokens with low initial probability but high reward, once high-probability tokens have stabilized. For
instance, under the Pass@k metric, raising the probability of at least one correct action above 1/k corresponds to an
expansion of the reasoning capability boundary.
3.2
Demonstration with A Toy Example
Next, to more clearly demonstrate the theoretically predicted two-stage dynamic, we validate the above analysis of
probability mass dynamics using a simple toy setting, and subsequently review several widely adopted RLVR tricks and
more than three actions case in Appendix C.4.
2Without loss of generality, we approximate importance ratio w ≈1, as regularization components such as the KL penalty and
clipping operation are applied in practical training.
5


## Page 6

A preprint
⋄Starting with a toy setting. To better track the probability mass dynamics, we analyze the scenario in a clean and
simplified setting, assuming the entire action space consists of only three actions3: a1, with r(a1) > 0, which has been
explored; a2, with r(a2) > 0, which remains unexplored; and a3, with r(a3) = 0, which has been explored. Let the
initial logits be denoted as z = [z(a1), z(a2), z(a3)]T , and the policy as π(ai) = exp (z(ai))/ P
j̸=i exp (z(aj)), ∀i ∈
[1, 2, 3]. Here, we perform G action samplings, estimate the relative advantages via ˆA(ai) = r(ai)−mean({r(aj)}G
j=1),
and subsequently update the logits z using the policy gradient update rule given in Theorem 1. As stated in Remark
3, we discuss the following scenario4: RLVR reinforces high-probability yet suboptimal actions while overlooking
potentially optimal correct actions that initially have low probability, thereby leading to over-exploitation behavior.
That is, r(a1) < r(a2), while initially π(a1) > π(a2). For implementation details, see Algorithm 1.
Results for demonstration. We sample actions at each optimization step (with G = 2, η = 0.1) and analyze
the probability mass dynamics (a larger G leads to more stable optimization and does not affect our main findings
and conclusions). Figure 1 clearly illustrates the dynamics of the probability mass of policy optimization across
different rewards r & initial policy probabilities π, which aligns with the theoretical analysis in Section 3.1. That
is, E(∆z(ai)) = ηπ(ai)
h
(1 −π(ai)) ˆA(ai) −P3
j̸=i π(aj) ˆA(aj)
i
. More specifically, the overall dynamics can be
divided into two stages: (1) Initially, π(a1) and π(a3) are relatively large while π(a2) is comparatively small. Since
actions a1 and a3 are predominantly sampled, and given that ˆA(a1) > 0 and ˆA(a3) < 0, π(a1) increases while π(a3)
decreases. Meanwhile, π(a2) remains almost unchanged. (2) As π(a1) increases, the gradient term 1 −π(a1) gradually
approaches zero, causing the growth of π(a1) to stabilize. If training continues beyond this point, when action a2
is sampled with ˆA(a2) > 0 and ˆA(a1) < 0, π(a1) will decrease while π(a2) increases. Note that throughout the
optimization process, the relative negative actions change (initially a3 and later a1).
Remark 4. From the two-stage dynamics, (1) it can be observed that although the relative policy gradient method
does exhibit the phenomenon of capability boundary shrinkage. However, prolonging the duration of the training
may result in further gradient updates being applied to low-probability action sequences once the high-probability
ones have reached convergence. This is precisely why the research represented by Cui et al. [12] employs entropy
control mechanisms to extend the duration of training. (2) More interestingly, the relative policy gradient may undergo
changes during the training process: π(a1) first increases and then decreases. Therefore, simply using the momentum
of policy gradients from the early stages of updates—as in methods like AAPO [69]—to enhance policy optimization
is suboptimal. In contrast, approaches such as ProRL [37, 36] periodically reset the reference policy and optimizer
states during training.
4
How to Prolong Training: Revisiting the Role of Relative Negative Gradients
Thus far, we have established the imperative of avoiding over-sharpening in the policy distribution—which induces
over-exploitation and entropy collapse—and of enabling sustained training. Liu et al. [37] identify a fundamental
limitation across existing studies [78, 13, 83]: RL training is frequently terminated prematurely after only a few hundred
steps, hindering the models’ ability to fully explore and acquire novel reasoning capabilities. Their conclusions align
closely with our findings. Therefore, enhancing training stability and facilitating extended training durations constitute
promising directions for future research.
In group policy optimization (e.g. GRPO, GSPO), the policy πθ(y | x) = Q|y|
t=1 πθ(yt | x, y<t) learned by the model is
inherently complex. Returning to Section 3, policy updates can be interpreted as dynamically redistributing probability
mass across the search tree, which has a size of O(V T ). To unlock the model’s capacity for genuinely novel reasoning,
we call for research into strategies that more effectively allocate probability mass. Based on the probability mass
dynamics established in Lemma 1 and Theorem 1, optimizing relative negative advantage actions implicitly increases
the probability of other actions. A straightforward strategy is to allocate policy probability mass exclusively through
relative negative gradients within the overall dynamics. In this part, we will revisit the role of only using relative
negative gradients in prolonging training.
4.1
Experimental Setup
We choose Qwen2.5-Math-7B [72] and Llama-3.2-3B-Instruct [59] as our base models for investigation, which
align with our hardware resource. For RLVR algorithms, we evaluate the standard approach alongside a variant that
3In Section 2, the reward r is sparse (1 or 0). However, due to the presence of factors such as the importance ratio w, distinctions
arise among positive rewards (wr). For brevity of analysis, we ignore the standard deviation std({r(aj)}G
j=1) because it does not
affect the sign (positive or negative nature) of ˆA(ai).
4Otherwise, we can proceed with normal optimization to increase the probability of the optimal action.
6


## Page 7

A preprint
employs exclusively relative negative gradients5 (denoted as -N). This comparison includes widely-used methods such
as GRPO [21, 54] and GSPO [86]. Moreover, we use the verl framework [55] to train the models and the detailed
hyperparameter settings of training and evaluation can be found in Appendix C.2. For the datasets, we employ the
training set of MATH [27], which comprises 7,500 problems, for model training (with prompt batch size of 1,024).
Performance is evaluated on widely-used reasoning benchmarks, (1) in-domain (ID) tasks: the test sets of MATH,
AIME 2024, AIME 2025, and AMC 2023. (2) out-of-domain (OOD) tasks: ARC-c [11] (open-domain reasoning),
MMLU-Pro [62] (academic reasoning).
Specifically, we adopt Pass@k as our primary evaluation metric, which measures whether a model can successfully solve
a problem within k attempts. This metric has been widely used to mitigate the unreliability of greedy decoding-based
accuracy estimates [28] and to better assess the true capability boundaries of models [7, 9, 78, 88]. The unbiased
estimator first generates the n responses for per question x (n ≥k), counts the number of correct responses c, then
computes the metric as:
Pass@k = Ex∼D
"
1 −
 n−c
k

 n
k

#
.
4.2
Training Dynamics and Evaluation Results
0
6
12
18
24
30
36
42
0.64
0.66
0.68
0.70
0.72
0.74
0.76
0.78
Accuracy
MATH Test Score
GRPO
GRPO-N
GSPO
GSPO-N
0
6
12
18
24
30
36
42
0.08
0.09
0.10
0.11
0.12
0.13
0.14
0.15
0.16
Entropy
MATH Test Entropy
GRPO
GRPO-N
GSPO
GSPO-N
0
6
12
18
24
30
36
42
0.10
0.15
0.20
0.25
0.30
0.35
0.40
Value
Actor Entropy Loss
GRPO
GRPO-N
GSPO
GSPO-N
0
6
12
18
24
30
36
42
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
Value
Critic Rewards Mean
GRPO
GRPO-N
GSPO
GSPO-N
Figure 2: Comparison of the training dynamics of GRPO, GRPO-N, GSPO, and GSPO-N on the MATH benchmark
across training steps, using the Qwen2.5-Math-7B model with a prompt batch size of 1,024. Left Part: (Left) the
greedy decoding accuracy on the MATH test set and (Center Left) the model’s entropy on the MATH test set. Right
Part: (Center Right) the actor entropy loss and (Right) critic rewards mean during training. GRPO causes the entropy of
the base model to collapse over the course of training, suggesting a loss of exploratory capability. In contrast, GRPO-N,
GSPO, and GSPO-N all exhibit a pattern where entropy initially decreases and then increases. Notably, the entropy of
GRPO-N significantly surpasses that of the base model. All algorithms achieve competitive performance in both greedy
decoding accuracy and critic rewards mean.
Training dynamics.
We characterize the training dynamics by monitoring the greedy decoding accuracy and
entropy on a held-out MATH test set over the course of training (Figure 2 for Qwen2.5-Math-7B, Figure 4 for
Llama-3.2-3B-Instruct), together with the actor entropy loss and critic rewards mean during training. As illus-
trated, GRPO, GRPO-N, GSPO, and GSPO consistently achieve competitive performance in both greedy decoding
accuracy and critic rewards mean. Notably, GRPO leads to a rapid and substantial decline in entropy on the MATH
test set. In contrast, GRPO-N, GSPO, and GSPO-N all show an initial decrease in entropy, followed by a consistent
increase. Importantly, the entropy on the held-out test set under GRPO-N significantly exceeds that of the base model.
This divergence indicates that the standard GRPO may limit output diversity and exploratory capability (see Table 1),
both methods that apply sequence-level importance ratio clipping directly (GSPO and GSPO-N) and those that utilize
only relative negative gradients (GRPO-N) help mitigate overconfidence in previously sampled responses. Of particular
significance, prior study [12] suggests that policy performance comes at the cost of policy entropy, and is therefore
bottlenecked by its exhaustion. Therefore, the model optimized by GRPO-N may be a good baseline and maintain the
base model’s diversity for prolonging training6.
Performance on ID&OOD tasks. As shown in Table 1, for model with strong prior (e.g., Qwen models), both
GRPO-N and GSPO-N consistently achieve a favorable trade-off across various values of k on both ID tasks (e.g., AMC
2023, AIME 2024, and AIME 2025) and OOD tasks (such as ARC-c and MMLU-Pro). In particular, (1) GSPO-N
5The relative advantage ˆAi,t is computed over a group of responses. When ˆAi,t < 0, the term wi,t(θ) ˆAi,t∇θ log πθ(yi,t |
x, yi,<t) is referred to as a relative negative gradient. See Appendix B.5 for the details of relative negative gradients.
6For model with weak prior, we provide the training dynamics and evaluation results of LLama-3.2-3B-Instruct in Appendix
C.3. The performance ceiling is related to the base model, yet the key finding remains consistent across different models.
7


## Page 8

A preprint
matches the best Pass@1 performance on AMC 2023, AIME 2025, ARC-c and MMLU-Pro. (2) GRPO-N and GSPO-N
reliably improve the reasoning performance of the base model on ID tasks for every value of k. (3) For OOD tasks,
GRPO-N (GSPO-N) achieves higher Pass@k scores than GRPO (GSPO) across all k values, demonstrating stable
performance improvements while largely preserving the diversity of the base model.
Table 1: Evaluation results of Qwen2.5-Math-7B on in-domain tasks (AMC 2023, AIME 2024, and AIME 2025)
and out-of-domain tasks (ARC-c and MMLU-Pro). For each k, bold and underlined numbers indicate the best and
second-best results, respectively.
Algorithm
Pass@k
k
1
2
4
8
16
32
64
128
256
AMC 2023
Base Model
40.4
55.6
69.1
79.4
85.9
89.5
92.1
94.6
97.5
GRPO
60.4
69.9
77.4
82.9
86.7
89.4
91.7
94.7
97.5
GRPO-N
59.2
68.7
76.3
82.7
87.6
92.3
96.3
99.1
100.0
GSPO
61.1
70.5
78.0
83.9
88.1
91.6
94.4
96.2
97.5
GSPO-N
61.5
71.2
78.5
84.1
88.4
91.8
94.8
97.4
100.0
AIME 2024
Base Model
13.6
21.8
30.5
37.5
43.5
49.7
55.8
61.4
66.7
GRPO
22.6
31.5
39.5
46.2
51.9
57.3
62.9
68.9
73.3
GRPO-N
23.6
33.4
41.8
47.5
51.9
56.7
61.8
67.3
73.3
GSPO
25.3
34.7
42.4
48.3
53.6
58.7
63.6
68.1
73.3
GSPO-N
23.3
31.1
42.1
48.8
54.3
59.4
64.4
69.6
73.3
AIME 2025
Base Model
6.4
10.2
14.5
18.9
23.6
28.1
32.5
38.3
46.7
GRPO
9.2
13.4
17.9
22.4
26.4
29.9
33.9
39.1
46.7
GRPO-N
9.5
14.2
19.1
23.8
28.7
34.2
41.6
52.5
66.7
GSPO
9.6
14.3
19.2
23.9
28.8
34.2
40.9
49.5
60.0
GSPO-N
10.2
14.7
19.6
24.9
29.9
35.0
40.9
47.3
53.3
ARC-c
Base Model
35.4
54.9
73.7
86.5
93.6
96.9
98.2
99.2
100.0
GRPO
62.3
77.4
86.6
91.6
94.2
96.3
98.2
99.5
100.0
GRPO-N
61.7
78.1
88.5
94.3
97.7
99.5
99.9
100.0
100.0
GSPO
59.9
74.1
83.2
89.1
93.2
95.7
96.7
96.9
96.9
GSPO-N
63.9
77.9
86.5
91.1
93.7
95.5
96.5
96.9
96.9
MMLU-Pro
Base Model
28.1
41.4
55.1
67.6
78.1
85.9
91.6
96.2
100.0
GRPO
40.1
52.0
62.4
70.8
76.9
80.6
83.7
87.3
90.6
GRPO-N
38.5
49.9
60.7
70.1
78.1
84.3
89.0
93.7
100.0
GSPO
40.0
50.6
60.6
69.2
75.5
79.5
82.4
85.9
90.6
GSPO-N
41.6
52.3
61.8
69.7
75.9
79.9
83.6
88.1
93.8
In this subsection, we demonstrate that while all algorithms achieve competitive performance, they exhibit distinct
behaviors at the entropy level. This observation motivates further analysis of the differences among various correct
responses in the next part.
4.3
Analysis of Different Correct Responses
Case study. We present the reasoning processes of GRPO and GRPO-N, as illustrated in Figure 3. Although
various responses can yield correct answers through reasoning, their underlying reasoning behaviors differ markedly.
Specifically, GRPO tends to repeat similar errors, such as persistently generating code with TypeErrors, and fails to
rectify them. In contrast, GRPO-N produces fewer erroneous codes than GRPO. We further examined the responses
generated during the training process and found instances where incorrect code was initially produced but later reflected
upon and corrected to form the final correct response. GRPO assigns higher probability to the entire trajectory (i.e., the
behavior of generating incorrect code is reinforced—a tendency that may significantly affect the model’s self-correction
ability), whereas GRPO-N mitigates this issue. This necessitates an inquiry into achieving finer-grained control of
probability assignments.
8


## Page 9

A preprint
Question: Consider the set of complex numbers 𝑧satisfying |1 + 𝑧+ 𝑧2| = 4. The maximum value of the imaginary 
part of 𝑧can be written in the form 𝑚
n , where 𝑚and 𝑛are relatively prime positive integers. What is 𝑚+ 𝑛?
Ground-truth Answer: 21  
Let's break down the problem… 1. Let 𝑧= 𝑥+ 𝑦𝑖, where 𝑥and 𝑦are real 
numbers. 2. Substitute 𝑧and 𝑧2 into the given equation, and squaring both 
sides: 1 + 𝑥+ 𝑥2 −𝑦2 2 +
𝑦+ 2𝑥𝑦2 = 16. 3. We need to find the 
maximum value of 𝑦that satisfies the above equation. Let's use Python and 
SymPy to solve this. <…Python code…> < …TypeError… > 4. The error 
occurs because……  revised code: <…Python code…> < …TypeError … >  
5. < Reproduce revised code behavior…> 6. The numerical approach can be 
computationally intensive and may not be the most efficient way to solve this 
problem. Instead, let's use a more analytical approach…… 7. By analyzing the 
equation, we can see that the maximum value of 𝑦occurs when the real part of 
the expression inside the magnitude is minimized and the imaginary part is 
maximized. This happens when 𝑥= −1/2 which simplifies the equation to:
1 −1
2 +
−1
2
2
−𝑦2
2
+
𝑦+ 2
−1
2 𝑦
2
= 16
8. Taking the square root of both sides, we get: 
3
4 −𝑦2 = 4. Since 𝑦2 cannot 
be negative. Solving the second case: 
3
4 −𝑦2 = −4. 𝑦= ±
19
2
. The values 
𝑚and 𝑛are 19 and 2, respectively…... Therefore, 𝑚+ 𝑛= 19 + 2 = 21.
The final answer is:
\[
\boxed{21}
\]
Let's break down the problem… 1. Let 𝑧= 𝑥+ 𝑦𝑖, where 𝑥and 
𝑦are real numbers. 2. Square both sides to remove the square root: 
1 + 𝑥+ 𝑥2 −𝑦2 2 +
𝑦+ 2𝑥𝑦2 = 16. 3. Let's proceed with 
a numerical approach using Python to find the maximum value of 𝑦. 
Let's write the Python code to perform this search. <…Python 
code…><…Output…> 4. The numerical method suggests that the 
maximum value of 𝑦is approximately −0.03, which is very close to 0. 
This indicates that the maximum imaginary part of 𝑧is likely 0, but 
let's verify this by checking the exact form of the solution. 5. Let's 
consider the equation again: 1 + 𝑥+ 𝑥2 −𝑦2 2 + ሺ
ሻ
𝑦+
2𝑥𝑦2 = 16. 6. We can see that the maximum imaginary part occurs 
when 𝑥= −1/2. Substituting 𝑥= −1/2 into the equation, we get:
1 −1
2 + 1
4 −𝑦2
2
+
𝑦−𝑦2 =
3
4 −𝑦2
2
= 16
7. Solving for 𝑦, we get: 𝑦2 =
3
4 𝑥± 4. Since 𝑦2 must be non-negative, 
we take the positive root: 𝑦= ±
19
2 . Therefore, 𝑚= 19 and 𝑛= 2, 
so 𝑚+ 𝑛= 21. Thus, the final answer is:
\[
\boxed{21}
\]
GRPO-N
GRPO
Figure 3: A comparison of the correct responses of GRPO and GRPO-N (a test case from AMC 2023). The key
reasoning steps are presented here, see Appendix C.6 for full procedure.
5
Conclusion and Discussion
Conclusion. Based on a two-stage dynamic view of probability mass allocation, this study resolves the ongoing debate
on whether RLVR shrinks or expands LLM reasoning capabilities. We show that initial training favors exploitation,
potentially narrowing capability boundaries, while prolonged training encourages exploration, enabling genuine
expansion. Theoretically and empirically, we demonstrate that both phenomena occur at different phases. Guided by
these findings, one can develop new algorithms to foster more advanced reasoning capabilities.
Discussion. However, further studies are required on (i) how to design efficient algorithms for fine-grained probability
mass allocation; (ii) what kind of base models are more conducive to capability boundary expansion during the RL
stage; and (iii) where the ceiling of boundary exploration lies. We leave these questions for our future work.
References
[1] Abbas Abdolmaleki, Bilal Piot, Bobak Shahriari, Jost Tobias Springenberg, Tim Hertweck, Michael Bloesch,
Rishabh Joshi, Thomas Lampe, Junhyuk Oh, Nicolas Heess, Jonas Buchli, and Martin Riedmiller. Learning from
negative feedback, or positive feedback or both. In International Conference on Learning Representations, 2025.
[2] Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu,
Mingxuan Wang, and Lingpeng Kong. Polaris: A post-training recipe for scaling reinforcement learning on
advanced reasoning models, 2025. URL https://hkunlp.github.io/blog/2025/Polaris.
[3] Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Russ R Salakhutdinov, and Ruosong Wang. On exact
computation with an infinitely wide neural net. Advances in neural information processing systems, 32, 2019.
[4] Chenjia Bai, Yang Zhang, Shuang Qiu, Qiaosheng Zhang, Kang Xu, and Xuelong Li. Online preference alignment
for language models via count-based exploration. In The Thirteenth International Conference on Learning
Representations, 2025.
[5] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav
Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning
from human feedback. arXiv preprint arXiv:2204.05862, 2022.
9


## Page 10

A preprint
[6] Hongyi James Cai, Junlin Wang, Xiaoyin Chen, and Bhuwan Dhingra. How much backtracking is enough?
exploring the interplay of sft and rl in enhancing llm reasoning. arXiv preprint arXiv:2505.24273, 2025.
[7] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri
Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code.
arXiv preprint arXiv:2107.03374, 2021.
[8] Peter Chen, Xiaopeng Li, Ziniu Li, Xi Chen, and Tianyi Lin. Spectral policy optimization: Coloring your incorrect
reasoning in grpo. arXiv preprint arXiv:2505.11595, 2025.
[9] Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, and Guang Shi. Pass@ k training
for adaptively balancing exploration and exploitation of large reasoning models. arXiv preprint arXiv:2508.10751,
2025.
[10] Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement
learning baseline for model reasoning. arXiv preprint arXiv:2504.02546, 2025.
[11] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.
Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457,
2018.
[12] Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu
Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv
preprint arXiv:2505.22617, 2025.
[13] Xingyu Dang, Christina Baek, J Zico Kolter, and Aditi Raghunathan. Assessing diversity collapse in reasoning.
In Scaling Self-Improving Foundation Models without Human Supervision, 2025.
[14] Xingyu Dang, Christina Baek, Kaiyue Wen, J Zico Kolter, and Aditi Raghunathan. Weight ensembling improves
reasoning in language models. In Second Conference on Language Modeling, 2025.
[15] Wenlong Deng, Yi Ren, Muchen Li, Danica J Sutherland, Xiaoxiao Li, and Christos Thrampoulidis. On the effect
of negative gradient in group relative deep reinforcement optimization. arXiv preprint arXiv:2505.18830, 2025.
[16] Zhun Deng, Hangfeng He, and Weijie Su. Toward better generalization bounds with locally elastic stability. In
International Conference on Machine Learning, pages 2590–2600, 2021.
[17] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang,
KaShun SHUM, and Tong Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment.
Transactions on Machine Learning Research, 2023. ISSN 2835-8856.
[18] Yihong Dong, Xue Jiang, Yongding Tao, Huanyu Liu, Kechi Zhang, Lili Mou, Rongyu Cao, Yingwei Ma, Jue
Chen, Binhua Li, Zhi Jin, Fei Huang, Yongbin Li, and Ge Li. Rl-plus: Countering capability boundary collapse of
llms in reinforcement learning with hybrid-policy optimization. arXiv preprint arXiv:2508.00222, 2025.
[19] Stanislav Fort, Paweł Krzysztof Nowak, Stanislaw Jastrzebski, and Srini Narayanan. Stiffness: A new perspective
on generalization in neural networks. arXiv preprint arXiv:1901.09491, 2019.
[20] Kanishk Gandhi, Ayush K Chakravarthy, Anikait Singh, Nathan Lile, and Noah Goodman. Cognitive behaviors
that enable self-improving reasoners, or, four habits of highly effective STars. In Second Conference on Language
Modeling, 2025.
[21] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi
Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv
preprint arXiv:2501.12948, 2025.
[22] Shangmin Guo, Yi Ren, Stefano V Albrecht, and Kenny Smith. lpNTK: Better generalisation with less data via
sample interaction during learning. In The Twelfth International Conference on Learning Representations, 2024.
[23] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training
large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.
[24] Andre He, Daniel Fried, and Sean Welleck. Rewarding the unlikely: Lifting grpo beyond distribution sharpening.
arXiv preprint arXiv:2506.02355, 2025.
10


## Page 11

A preprint
[25] Hangfeng He and Weijie Su. The local elasticity of neural networks. In International Conference on Learning
Representations, 2020.
[26] Zhiyuan He, Xufang Luo, Yike Zhang, Yuqing Yang, and Lili Qiu. δl normalization: Rethink loss aggregation in
rlvr. arXiv preprint arXiv:2509.07558, 2025.
[27] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob
Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on
Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.
[28] Andreas Hochlehnert, Hardik Bhatnagar, Vishaal Udandarao, Samuel Albanie, Ameya Prabhu, and Matthias
Bethge. A sober look at progress in language model reasoning: Pitfalls and paths to reproducibility. arXiv preprint
arXiv:2504.07086, 2025.
[29] Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. Reinforce++: An efficient rlhf algorithm with robustness to
both prompt and reward models. arXiv preprint arXiv:2501.03262, 2025.
[30] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero:
An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290,
2025.
[31] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in
neural networks. Advances in neural information processing systems, 31, 2018.
[32] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander
Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
[33] Chengao Li, Hanyu Zhang, Yunkun Xu, Hongyan Xue, Xiang Ao, and Qing He. Gradient-adaptive policy
optimization: Towards multi-objective alignment of large language models. arXiv preprint arXiv:2507.01915,
2025.
[34] Jiacai Liu. Brief introduction of policy gradient in llm reasoning. https://notion.so/Brief-Introduction-of-Policy-
Gradient-In-LLM-Reasoning-1c04795a3e8b805abbd6ccc9f1a34ac0LiuLiu, 2025.
[35] Jiawei Liu and Lingming Zhang. Code-r1: Reproducing r1 for code with reliable rewards. arXiv preprint
arXiv:2503.18470, 3, 2025.
[36] Mingjie Liu, Shizhe Diao, Jian Hu, Ximing Lu, Xin Dong, Hao Zhang, Alexander Bukharin, Shaokun Zhang,
Jiaqi Zeng, Makesh Narsimhan Sreedhar, et al. Scaling up rl: Unlocking diverse reasoning in llms via prolonged
training. arXiv preprint arXiv:2507.12507, 2025.
[37] Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged
reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864,
2025.
[38] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin.
Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
[39] Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin,
Colin Cai, Maurice Weber, et al. Deepcoder: A fully open-source 14b coder at o3-mini level. Notion Blog, 2025.
[40] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y Tang, Manan Roongta, Colin Cai, Jeffrey Luo,
Tianjun Zhang, Li Erran Li, et al. Deepscaler: Surpassing o1-preview with a 1.5 b model by scaling rl. Notion
Blog, 2025.
[41] Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen,
Runming He, Bin Cui, and Wentao Zhang. Learning what reinforcement learning can’t: Interleaved online
fine-tuning for hardest questions. arXiv preprint arXiv:2506.07527, 2025.
[42] Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen,
Runming He, Bin Cui, et al. Learning what reinforcement learning can’t: Interleaved online fine-tuning for hardest
questions. arXiv preprint arXiv:2506.07527, 2025.
11


## Page 12

A preprint
[43] Laura O’Mahony, Leo Grinsztajn, Hailey Schoelkopf, and Stella Biderman. Attributing mode collapse in the
fine-tuning of large language models. In ICLR 2024 Workshop on Mathematical and Empirical Understanding of
Foundation Models, 2024.
[44] OpenAI. Introducing openai o3 and o4-mini, 2025. Accessed: April 16, 2025.
[45] Shunyu Yao (OpenAI). The second half. https://ysymyth.github.io/The-Second-Half/, 2025.
[46] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human
feedback. Advances in neural information processing systems, pages 27730–27744, 2022.
[47] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by
tracing gradient descent. Advances in Neural Information Processing Systems, 33:19920–19930, 2020.
[48] Chen Qian, Dongrui Liu, Haochen Wen, Zhen Bai, Yong Liu, and Jing Shao. Demystifying reasoning dy-
namics with mutual information: Thinking tokens are information peaks in llm reasoning. arXiv preprint
arXiv:2506.02867, 2025.
[49] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct
preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290,
2024.
[50] Yi Ren and Danica J Sutherland. Learning dynamics of llm finetuning. In International Conference on Learning
Representations, 2025.
[51] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous
control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.
[52] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization
algorithms. arXiv preprint arXiv:1707.06347, 2017.
[53] Darsh J Shah, Peter Rushton, Somanshu Singla, Mohit Parmar, Kurt Smith, Yash Vanjani, Ashish Vaswani,
Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, et al. Rethinking reflection in pre-training. arXiv preprint
arXiv:2504.04022, 2025.
[54] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang,
YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models.
arXiv preprint arXiv:2402.03300, 2024.
[55] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin,
and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European
Conference on Computer Systems, pages 1279–1297, 2025.
[56] Yuda Song, Hanlin Zhang, Carson Eisenach, Sham M. Kakade, Dean Foster, and Udaya Ghai. Mind the gap:
Examining the self-improvement capabilities of large language models. In The Thirteenth International Conference
on Learning Representations, 2025.
[57] Yiyou Sun, Yuhan Cao, Pohao Huang, Haoyue Bai, Hannaneh Hajishirzi, Nouha Dziri, and Dawn Song. Delta-
code: How does rl unlock and transfer new programming algorithms in llms? arXiv preprint arXiv:2509.21016,
2025.
[58] Gokul Swamy, Sanjiban Choudhury, Wen Sun, Zhiwei Steven Wu, and J Andrew Bagnell. All roads lead to
likelihood: The value of reinforcement learning in fine-tuning. arXiv preprint arXiv:2503.01067, 2025.
[59] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language
models. arXiv preprint arXiv:2302.13971, 2023.
[60] Luong Trung, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. ReFT: Reasoning with reinforced
fine-tuning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7601–7614, 2024.
[61] Haozhe Wang, Qixin Xu, Che Liu, Junhong Wu, Fangzhen Lin, and Wenhu Chen. Emergent hierarchical reasoning
in llms through reinforcement learning. arXiv preprint arXiv:2509.03646, 2025.
12


## Page 13

A preprint
[62] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran
Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding
benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.
[63] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.
Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing
systems, 35:24824–24837, 2022.
[64] Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu, Shengyu Ye, Zhirong Wu, Xiao Liang, Yang Wang, Junjie Li,
Ziming Miao, et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in
base llms. arXiv preprint arXiv:2506.14245, 2025.
[65] Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning.
Mach. Learn., page 229–256, 1992. ISSN 0885-6125.
[66] Fang Wu, Weihao Xuan, Ximing Lu, Zaid Harchaoui, and Yejin Choi. The invisible leash: Why rlvr may not
escape its origin. arXiv preprint arXiv:2507.14843, 2025.
[67] Jinyang Wu, Chonghua Liao, Mingkuan Feng, Shuai Zhang, Zhengqi Wen, Pengpeng Shao, Huazhe Xu, and
Jianhua Tao. Thought-augmented policy optimization: Bridging external guidance and internal capabilities. arXiv
preprint arXiv:2505.15692, 2025.
[68] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. Less: Selecting influential
data for targeted instruction tuning. arXiv preprint arXiv:2402.04333, 2024.
[69] Jian Xiong, Jingbo Zhou, Jingyong Ye, and Dejing Dou. Aapo: Enhance the reasoning capabilities of llms with
advantage momentum. arXiv preprint arXiv:2505.14264, 2025.
[70] Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming
Xiong, et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint
arXiv:2504.11343, 2025.
[71] Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason
under off-policy guidance. arXiv preprint arXiv:2504.14945, 2025.
[72] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu,
Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via
self-improvement. arXiv preprint arXiv:2409.12122, 2024.
[73] Zhaohui Yang, Yuxiao Ye, Shilei Jiang, Chen Hu, Linjing Li, Shihong Deng, and Daxin Jiang. Unearthing
gems from stones: Policy optimization with negative sample augmentation for llm reasoning. arXiv preprint
arXiv:2505.14403, 2025.
[74] Zhihe Yang, Xufang Luo, Zilong Wang, Dongqi Han, Zhiyuan He, Dongsheng Li, and Yunjian Xu. Do not let
low-probability tokens over-dominate in rl for llms. arXiv preprint arXiv:2505.12929, 2025.
[75] Jiahao Yu, Zelei Cheng, Xian Wu, and Xinyu Xing. Gpo: Learning from critical steps to improve llm reasoning.
arXiv preprint arXiv:2509.16456, 2025.
[76] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong
Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint
arXiv:2503.14476, 2025.
[77] Lifan Yuan, Weize Chen, Yuchen Zhang, Ganqu Cui, Hanbin Wang, Ziming You, Ning Ding, Zhiyuan Liu,
Maosong Sun, and Hao Peng. From f(x) and g(x) to f(g(x)): LLMs learn new skills in RL by composing
old ones. https://husky-morocco-f72.notion.site/From-f-x-and-g-x-to-f-g-x-LLMs-Learn-New-Skills-in-RL-by-
Composing-Old-Ones-2499aba4486f802c8108e76a12af3020, 2025. Notion blog post, available online.
[78] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does
reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In 2nd AI for Math
Workshop @ ICML 2025, 2025.
[79] Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian
Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv
preprint arXiv:2504.05118, 2025.
13


## Page 14

A preprint
[80] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investi-
gating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892,
2025.
[81] Chuheng Zhang, Wei Shen, Li Zhao, Xuyun Zhang, Xiaolong Xu, Wanchun Dou, and Jiang Bian. Policy filtration
for RLHF to mitigate noise in reward models. In Forty-second International Conference on Machine Learning,
2025.
[82] Kaichen Zhang, Yuzhong Hong, Junwei Bao, Hongfei Jiang, Yang Song, Dingqian Hong, and Hui Xiong. Gvpo:
Group variance policy optimization for large language model post-training. arXiv preprint arXiv:2504.19599,
2025.
[83] Rosie Zhao, Alexandru Meterez, Sham M. Kakade, Cengiz Pehlevan, Samy Jelassi, and Eran Malach. Echo
chamber: RL post-training amplifies behaviors learned in pretraining. In Second Conference on Language
Modeling, 2025.
[84] Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reason without
external rewards. arXiv preprint arXiv:2505.19590, 2025.
[85] Chujie Zheng, Pei Ke, Zheng Zhang, and Minlie Huang. Click: Controllable text generation with sequence
likelihood contrastive learning. In Findings of the Association for Computational Linguistics: ACL 2023, pages
1022–1040, 2023.
[86] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui
Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
[87] Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. Reasoning by superposition:
A theoretical perspective on chain of continuous thought. arXiv preprint arXiv:2505.12514, 2025.
[88] Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising effectiveness
of negative reinforcement in llm reasoning. arXiv preprint arXiv:2506.01347, 2025.
14


## Page 15

A preprint
A
More Related Works
Here, we discuss more related works to supplement the main text.
Reinforcement Learning for LLM Reasoning. Large language models (LLMs) are often post-trained using rein-
forcement learning (RL), both for preference alignment [46, 5] and to improve reasoning capabilities [54, 21]. Inspired
by Shao et al. [54], Liu [34] and Swamy et al. [58], this work reformulates methods like SFT, RFT [60], DPO, PPO,
and GRPO as maximum likelihood estimation governed by a Gradient Coefficient. This coefficient fundamentally
operates by amplifying gradients for favored responses and suppressing others, with its magnitude modulating the
preference intensity. Thus, the core challenge in policy gradient methods reduces to the accurate estimation of this
Gradient Coefficient (i.e., the advantage and importance ratio). For instance, AAPO [69] redefines advantage estimation
by incorporating advantage momentum. GAPO [33], GVPO [82] and ∆L Normalization [26] employ gradient normal-
ization to adaptively rescale each objective’s gradients, thereby finding a low-variance estimator. Meanwhile, Zhao
et al. [84] and Qian et al. [48] utilize a model’s own internal confidence measure (or entropy)—termed self-certainty to
improve reasoning skills. Additionally, hybrid approaches that integrate RL with SFT on external demonstration data
have been actively explored [6, 41, 71, 67, 18]. Despite these empirical advances, the fundamental question of whether
RLVR expands [37, 64, 36, 66, 77, 61, 4] or shrinks [78, 83, 56, 13, 24, 42, 53, 20, 14, 43] the reasoning capacities of
LLMs remains an open and actively debated issue. This is precisely what we aim to uncover.
LLM Learning Dynamics. Deep neural networks learn by adjusting their parameters through gradient descent. This
process, known as learning dynamics, connects how model predictions change to the gradients from individual training
examples. Learning dynamics prioritizes the analysis of a model’s relative training behavior over its convergence,
providing a means to assess the quality of individual training samples. To name a few, Pruthi et al. [47] introduce
“TracIn", a metric that measures how much a training example affects a model’s predictions, Xia et al. [68] later use it to
identify the most influential examples during instruction fine-tuning of LLMs. In a similar vein, Guo et al. [22] propose
a method based on the neural tangent kernel (NTK) regime to estimate the relative difficulty among different training
samples. Furthermore, Ren and Sutherland [50] highlight a unique “squeezing effect” to explain a previously observed
phenomenon in off-policy direct preference optimization (DPO [49]), where running DPO for too long makes even the
desired outputs less likely. Since RLVR methods—exemplified by PPO and GRPO—are on-policy and dynamically
evolving, we argue that analyzing learning dynamics can naturally offer a novel perspective for understanding the hot
debate (capability boundary shrinkage or expansion) in RLVR.
Gradient Analysis in Preference Optimization. DPO [49] has proven highly effective, as it relies solely on an offline
dataset of paired preference data. However, this reliance on paired data restricts its applicability in settings where only
unpaired feedback (e.g., solely positive or negative responses) is available. In response, Abdolmaleki et al. [1] introduce
a decoupled approach that independently controls the influence of positive and negative signals, enabling learning even
when only a single feedback type is available. Regarding online update methods, RAFT++ [17, 70]—a simple rejection
sampling approach utilizing only positively rewarded data—has been shown to deliver performance competitive with
GRPO. Conversely, Zhu et al. [88] report the surprising effectiveness of training exclusively on negatively rewarded
samples using REINFORCE [65], without reinforcing correct responses. As we demonstrate in the main text, the set of
samples considered “negative" is not static but evolves dynamically throughout optimization. It imperative to analyze
the underlying learning dynamics. In addition, Yang et al. [73] and Chen et al. [8] find that negative responses hold
learning value (e.g., self-reflection). However, existing methods overlook this by either discarding them (RFT) or
applying uniform penalties (RL), failing to leverage these nuanced signals. There are also some token-level gradient
analyses: Yang et al. [74] identify that RL training is skewed by low-probability tokens’ excessive gradient magnitudes,
impeding the learning from essential high-probability tokens; Deng et al. [15] empirically observe that GRPO can
suffer from what we call Lazy Likelihood Displacement: a failure to sufficiently increase, or even a decrease in, the
likelihood of correct answers during training. The above motivates us to analyze the expected update in RLVR, once
again emphasizing the essential role of fine-grained probability mass allocation.
15


## Page 16

A preprint
LLM Usage
Regarding the use of LLMs, they were employed solely for language polishing purposes and played no role in research
ideation, literature retrieval, or any other academically substantive activities.
B
Omitted Proofs and Additional Results
B.1
Proof of Equation 2
Proof. We begin by reviewing the objective function of GRPO below.
JGRPO(θ) =Ex∼D, {yi}G
i=1∼πθold(·|x)

1
G
G
X
i=1
1
|yi|
|yi|
X
t=1
n
min

wi,t(θ) ˆAi,t, clip
 wi,t(θ), 1 −ϵ, 1 + ϵ
 ˆAi,t
o
−βDKL(πθ || πref)

,
where wi,t(θ) =
πθ(yi,t|x,yi,<t)
πθold(yi,t|x,yi,<t), DKL(πθ || πref) = πref(yi,t|x,yi,<t)
πθ(yi,t|x,yi,<t) −log πref(yi,t|x,yi,<t)
πθ(yi,t|x,yi,<t) −1, β is the coefficient.
To better understand the model’s learning dynamics under this binary outcome reward setting, we omit the regularization
components (e.g., KL term & clipping operation):
JGRPO(θ) =Ex∼D, {yi}G
i=1∼πθold(·|x)

1
G
G
X
i=1
1
|yi|
|yi|
X
t=1
wi,t(θ) ˆAi,t

,
∇θJGRPO(θ) = Ex∼D, {yi}G
i=1∼πθold(·|x)

1
G
G
X
i=1
1
|yi|
|yi|
X
t=1
∇θwi,t(θ) ˆAi,t


= Ex∼D, {yi}G
i=1∼πθold(·|x)

1
G
G
X
i=1
1
|yi|
|yi|
X
t=1
∇θπθ(yi,t | x, yi,<t)
πθold(yi,t | x, yi,<t)
ˆAi,t


= Ex∼D, {yi}G
i=1∼πθold(·|x)

1
G
G
X
i=1
1
|yi|
|yi|
X
t=1
πθ(yi,t | x, yi,<t)
πθold(yi,t | x, yi,<t)
ˆAi,t∇θ log πθ(yi,t | x, yi,<t)


= Ex∼D, {yi}G
i=1∼πθold(·|x)

1
G
G
X
i=1
1
|yi|
|yi|
X
t=1
wi,t(θ) ˆAi,t
|
{z
}
coefficient
∇θ log πθ(yi,t | x, yi,<t)

.
We complete the proof of Equation 2. Notice that wi,t does not affect the sign of ˆAi,t.
Besides, one can also consider the gradient of the KL term (denote π(yi,t | x, yi,<t) as π(yi,t)):
∇θβDKL(πθ || πref) = β∇θ
πref(yi,t)
πθ(yi,t) −β∇θ log πref(yi,t)
πθ(yi,t)
= −β πref(yi,t)
π2
θ(yi,t) ∇θπθ(yi,t) + β∇θ log πθ(yi,t)
= −

β πref(yi,t)
πθ(yi,t) −β

∇θ log πθ(yi,t).
16


## Page 17

A preprint
B.2
Proof of Lemma 1
Proof. Re-stating the Lemma 1, the output of a model is the logits z = [z1, ..., zV ]T , which corresponds to a finite
(size V ) vocabulary set V = {v1, ..., vV }. The policy probability of the corresponding action (token) is calculated by:
π(v) = Softmax(z)v = exp (zv)/ PV
v′ exp (zv′).
That is, log π(v) = log(exp(zv)) −log(PV
v′ exp(z
′
v)) = zv −log(PV
v′ exp(z
′
v)).
Thus, for the currently sampled token v, let zv be its corresponding logit, we will have:
∂log π(v)
∂zv
= 1 −π(v),
for other unsampled tokens u ̸= v (with its logit zu):
∂log π(v)
∂zu
= −π(u).
Apply those to the gradient ∇zJ = ˆA(v)∇z log π(v), we complete the proof of Lemma 1.
B.3
Proposition 1 and Proof
Proposition 1. Let the conditions specified in Lemma 1 hold, and denote ∆z(x) = [∆z1, ..., ∆zV ]T , the l-th step
probability mass dynamics decompose as:
∆log πl(y | x) =

I −e(πl(y | x))T  h
(∇θzθl(x))(∇θzθl(x))T i
∆zl(x) + O(η2 ∇θzθl(x)

2
),
where I is the identity matrix and e = [1, 1, ..., 1]T ,
h
(∇θzθl(x))(∇θzθl(x))T i
∈RV ×V is the empirical neural
tangent kernel, ∆log πl(y | x) ∈RV ×1. ∆z(x) = η∇zJ ∈RV ×1, which mainly determines the direction and
magnitude of the policy update.
Proof. Recall the log probabilities change in Eq. (4):
∆log πl(y | x) ≜log πθl+1(y | x) −log πθl(y | x) := log πl+1(y | x) −log πl(y | x),
and we follow Ren and Sutherland [50] using Taylor expansion to approximate log πl+1(y | x):
log πl+1(y | x) = log πl(y | x) + ⟨∇log πl(y | x), θl+1 −θl⟩+ O(
θl+1 −θl2).
Then, supposing the parameters’ are updated by policy gradient, we will have (the model parameters θ ∈Rd×1):
∆log πl(y | x) = ∇θ log πl(y | x)(θl+1 −θl) + O(
θl+1 −θl2).
Next, we use the definition of gradient and the chain rule:
∇θ log πl(y | x)(θl+1 −θl) =
h
∇zθl log πl(y | x)(∇θzθl(x))
i
[η∇θlJ ]T
=
h
∇zθl log πl(y | x)(∇θzθl(x))
i h
η∇zθl J (∇θzθl(x))
iT
= ∇zθl log πl(y | x)
h
(∇θzθl(x))(∇θzθl(x))T i
(η∇zθl J )
=

I −e(πl(y | x))T  h
(∇θzθl(x))(∇θzθl(x))T i
∆zl(x).
For the higher-order term:
θl+1 −θl = η∇θlJ = η(∇θzθl(x))T ∇zθl J ,
and from the practical application and Lemma 1, the term ∇zθl J is usually bounded, we get:
O(
θl+1 −θl2) = O(η2 ∇θzθl(x)

2
).
We complete the proof.
h
(∇θzθl(x))(∇θzθl(x))T i
∈RV ×V denotes the empirical neural tangent kernel (NTK), which
remains nearly constant throughout the training process [50, 3, 31]. As a result, ∆zl(x) primarily governs both the
direction and magnitude of the policy update.
17


## Page 18

A preprint
B.4
Proof of Theorem 1
Theorem 1. Under the conditions stated in Lemma 1, we assume that x ∼D is i.i.d., the expected group relative policy
gradient ∇zJ ∈RV ×1 is Ex∼D,{ui}G
i=1∼π(·|x)
h
1
G
PG
i=1 ˆA(ui)∇z log π(ui)
i
. Then the expected logits update is:
E(∆zl
v) = η · Eu∼πl(·|x)
h
ˆA(u)∇zlv log πl(u)
i
= η · πl(v)

(1 −πl(v)) ˆA(v) −
X
u̸=v
πl(u) ˆA(u)

.
Proof. From Lemma 1, the policy gradient of sampling a token (action) u once from the policy πl(· | x) is
ˆA(u)∇z log π(u). Thus, the expected group relative policy gradient is the following:
∇zJ = Ex∼D,{ui}G
i=1∼π(·|x)
"
1
G
G
X
i=1
ˆA(ui)∇z log π(ui)
#
.
Given that x ∼D is i.i.d. and {ui}G
i=1 are randomly sampled from π(· | x), we derive an unbiased estimator:
∇zJ = Eu∼π(·|x)
h
ˆA(u)∇z log π(u)
i
=
X
u
π(u) ˆA(u)∇z log π(u) ∈RV ×1.
Apply those to Lemma 1, we complete the proof.
B.5
Details of Relative Negative Gradients
Referring back to Eq.(1) and Eq.(2), taking GRPO as an example, we obtain the gradient of the objective function in the
following form.
∇θJGRPO(θ) = Ex,{yi}G
i=1∼πθold(·|x)

1
G
G
X
i=1
1
|yi|
|yi|
X
t=1
wi,t(θ) ˆAi,t
|
{z
}
coefficient
∇θ log πθ(yi,t | x, yi,<t)

.
Since the advantage ˆAi,t is estimated from the currently sampled group i = 1, · · · , G each time, we refer to it as the
relative advantage, and correspondingly, this gradient is termed the relative policy gradient. Consequently, for the
relative negative gradients, we exclusively utilize gradient information where ˆAi,t < 0 during the gradient update:
∇θJGRPO-N(θ) = Ex,{yi}G
i=1∼πθold(·|x)

1
G
G
X
i=1
1
|yi|
|yi|
X
t=1
I( ˆAi,t) · wi,t(θ) ˆAi,t∇θ log πθ(yi,t | x, yi,<t)

,
where I( ˆAi,t) is an indicator variable that equals 1 if ˆAi,t < 0, and 0 otherwise.
18


## Page 19

A preprint
C
Extension to Experiments
Reproducibility statement. We employed open-source algorithms and data to validate our theoretical analysis, and
have reported all hyperparameter settings to facilitate reproducibility.
(1) open-source code: https://github.com/volcengine/verl.
(2) all datasets can be found in: https://huggingface.co/datasets.
(3) toy example details are provided in: Algorithm C.1.
C.1
Algorithm for Logits Update
Algorithm 1 Logits Update for Softmax Parameterization: A Toy Example
Require: learning rate η, number of samples per update G, true rewards r, optimization steps N
Initialize policy parameters (logits) z
for l = 1 to N do
Compute current policy π ←Softmax(z)
Sample G actions from policy π: {a1, a2, ..., aG}
Estimate advantage ˆA[ai] = r[ai] −mean({r[aj]}G
j=1)
Initialize relative policy gradient g ←0
for each sampled action ai where i = 1 to G do
g[ai] ←g[ai] + (1 −π[ai]) · ˆA[ai]
for each other action aj ̸= ai do
g[aj] ←g[aj] −π[aj] · ˆA[ai]
end for
end for
Apply Adam update: z ←z + η · g/G
end for
return Optimized policy parameters z
C.2
Hyperparameter Settings
Our experimental configuration follows that of Zhu et al. [88].
Training setup. The prompt batch size is set to 1,024, with 8 rollouts generated per prompt. During training, the sam-
pling temperature is set to 1.0. The maximum context length is configured as 4,096 tokens for both Qwen2.5-Math-7B
and Llama-3.2-3B-Instruct. Model updates are performed with a mini-batch size of 256 and a learning rate of
1 × 10−6. For all algorithms, a KL penalty term is incorporated into the final loss function, using a coefficient of
1 × 10−3. The clip ratio is set to 0.2. Additionally, an entropy bonus is applied to all objectives with a coefficient of
1 × 10−4. All experiments are conducted on a single node with 4 NVIDIA A100 GPUs.
Evaluation setup.
During evaluation, we sample 256 responses per prompt for both Qwen2.5-Math-7B and
Llama-3.2-3B-Instruct using a temperature of 0.6 and a top-p value of 0.95. Since the test sets of ARC-c
(1,170) and MMLU-Pro (12,000) are relatively large, and sampling 256 times requires substantial computation time, we
randomly selected 128 questions and repeated the test three times to obtain the average.
Prompt template. Our primary objective is to validate theoretical findings; therefore, a uniform prompt [80] was
sampled for all models:
<|im_start|>system
You are a helpful
assistant .<| im_end|>
<|im_start|>user
{input}
Please
reason
step by step , and put your
final
answer
within \boxed {}.
<|im_end|>
<|im_start|>assistant
19


## Page 20

A preprint
C.3
More Evaluation Results
Training dynamics of LLama-3.2-3B-Instruct.
0
6
12
18
24
30
36
42
0.48
0.49
0.50
0.51
0.52
0.53
0.54
0.55
Accuracy
MATH Test Score
GRPO
GRPO-N
0
6
12
18
24
30
36
42
0.130
0.135
0.140
0.145
0.150
0.155
0.160
0.165
0.170
Entropy
MATH Test Entropy
GRPO
GRPO-N
0
6
12
18
24
30
36
42
0.5
1.0
1.5
2.0
2.5
3.0
3.5
Value
Actor Entropy Loss
GRPO
GRPO-N
0
6
12
18
24
30
36
42
0.55
0.60
0.65
0.70
0.75
0.80
0.85
Value
Critic Rewards Mean
GRPO
GRPO-N
Figure 4: Comparison of the training dynamics of GRPO, GRPO-N on the MATH benchmark across training steps,
using the LLama-3.2-3B-Instruct model with a prompt batch size of 1,024. Left Part: (Left) the greedy decoding
accuracy on the MATH test set and (Center Left) the model’s entropy on the MATH test set. Right Part: (Center Right)
the actor entropy loss and (Right) critic rewards mean during training.
Table 2: Pass@k of Llama-3.2-3B-Instruct on AMC 2023, AIME 2024, AIME 2025. For each k, bold and
underlined numbers indicate the best and second-best results, respectively.
Algorithm
Pass@k
k
1
2
4
8
16
32
64
128
256
AMC 2023
Base Model
23.4
34.3
47.7
61.7
74.4
84.7
92.1
96.8
100.0
GRPO
31.1
41.7
51.3
58.7
64.7
70.7
76.9
83.0
87.5
GRPO-N
30.3
41.6
52.4
60.8
67.5
74.2
81.0
87.4
92.5
AIME 2024
Base Model
6.9
11.5
17.5
23.8
29.4
33.7
37.5
42.7
50.0
GRPO
15.7
20.6
25.1
29.1
32.2
34.4
36.1
37.9
40.0
GRPO-N
16.2
21.2
25.8
29.9
33.2
35.2
37.3
40.8
46.7
AIME 2025
Base Model
0.4
0.9
1.7
3.2
5.6
9.2
14.6
23.2
36.7
GRPO
0.6
1.1
2.1
3.8
6.2
9.0
11.7
14.4
16.7
GRPO-N
0.5
1.0
2.0
3.8
6.6
10.7
15.5
20.6
26.7
Table 3: Evaluation results of Qwen2.5-Math-7B on MATH-500. For each k, bold and underlined numbers indicate
the best and second-best results, respectively.
Algorithm
Pass@k
k
1
2
4
8
16
32
64
128
256
MATH-500
Base Model
40.7
51.5
58.9
64.1
68.5
72.9
77.9
83.2
88.0
GRPO
53.3
57.9
61.2
63.5
65.3
67.0
68.9
70.8
72.6
GRPO-N
53.0
57.9
61.3
63.7
65.3
66.8
68.5
70.2
72.2
GSPO
53.0
57.7
61.0
63.4
65.2
66.8
68.6
70.5
72.8
GSPO-N
54.1
58.8
62.0
64.1
65.9
67.6
69.5
71.5
73.4
20


## Page 21

A preprint
C.4
Discussion on RL Tricks
We also review some widely adopted RL tricks, such as: increasing the number of rollout samples, raising the training
temperature, more than three actions case.
• The number of rollout samples: a larger G leads to more stable optimization and does not affect our main findings
and conclusions, the two-stage dynamic of exploitation and exploration.
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.9
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.9
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.9
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.9
a2 : r2 = 1.0
a3 : r3 = 0
Figure 5: Dynamics of the policy probability mass during optimization for different numbers of rollout samples
([2, 3, 5, 10]), with action rewards r and initial policy probabilities π held constant.
• Raising the training temperature: according to An et al. [2], increasing the sampling temperature enhances the
diversity of generated outcomes. Consequently, employing a higher temperature is advisable to obtain a more varied
set of trajectories for model training. The default temperature value in our other experiments is τ = 1.0, that is
π(·) = Softmax(z/τ).
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.9
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.9
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.9
a2 : r2 = 1.0
a3 : r3 = 0
0
1000
2000
3000
4000
5000
6000
7000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.9
a2 : r2 = 1.0
a3 : r3 = 0
Figure 6: Dynamics of the policy probability mass during optimization for different training temperature values
([1, 2, 5, 20]), with action rewards r and initial policy probabilities π held constant.
•
More
than
three
actions
case:
from
Theorem
1,
we
have:
E(∆z(ai))
=
ηπ(ai)
h
(1 −π(ai)) ˆA(ai) −P4
j̸=i π(aj) ˆA(aj)
i
.
Denote the action with the largest r as amax and the action with the smallest r as amin. It can be readily shown that
E(∆z(amax)) is always greater than or equal to 0, while E(∆z(amin)) is consistently less than 0. For other actions, the
probabilities generally exhibit a two-stage dynamic.
0
100020003000400050006000700080009000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.9
a2 : r2 = 1.0
a3 : r3 = 0
a4 : r4 = 0.8
0
100020003000400050006000700080009000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.9
a2 : r2 = 1.0
a3 : r3 = 0
a4 : r4 = 0.85
0
100020003000400050006000700080009000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 1.0
a2 : r2 = 1.0
a3 : r3 = 0
a4 : r4 = 0.85
0
100020003000400050006000700080009000
Optimization Step
0.0
0.2
0.4
0.6
0.8
1.0
Probability (0-1)
a1 : r1 = 0.8
a2 : r2 = 1.0
a3 : r3 = 0
a4 : r4 = 0.7
Figure 7: Dynamics of the policy probability mass during optimization for action space consists of four actions, with
action rewards r and initial policy probabilities π held constant.
21


## Page 22

A preprint
C.5
Entropy Behavior Analysis from Different Levels
Set up. To investigate how different algorithms reshapes the sampling distribution, we compare the base model with the
RLVR trained model (using the experimental setup detailed in Section 4).
Following Wu et al. [66], we quantify changes in the output distribution using two entropy metrics:
• Answer-Level Entropy: Let {o(1), . . . , o(G)} represent the answers extracted from each generated sequence yi (with
NA denoting incomplete or invalid outputs), and let {o∗
1, . . . , o∗
M} be the set of M distinct answers. Denote by fj the
frequency of answer o∗
j, and define the empirical probability as pj = fj
G . The answer-level entropy is then defined
as: AnswerEntropy = −PM
j=1 pj log pj. This metric quantifies the global diversity across output completions, where
lower entropy values indicate a greater degree of answer-level certainty.
• Token-Level Entropy: Let V denote the vocabulary and yi = (yi,1, yi,2, . . . , yi,T ) denote the i-th generated sequence
of length T for 1 ≤i ≤N. At each timestep t, the model outputs a probability distribution p(i)
t (v) over vocabulary
tokens v ∈V. The entropy of this distribution is given by: H(p(i)
t ) = −P
v∈V p(i)
t (v) log p(i)
t (v). The average token-
level entropy over all G sequences and all timesteps is then computed as: TokenEntropy = 1
G
1
T
PG
i=1
PT
t=1 H(p(i)
t ).
Table 4: Summary of entropy metrics across math reasoning benchmarks.
Metric
Model
AMC 2023
AIME 2024
AIME 2025
Qwen2.5-Math-7B
2.563
4.263
4.904
GRPO
1.667
3.691
4.916
GRPO-N
1.741
3.748
4.957
Answer-Level
GSPO
1.641
3.484
4.878
Entropy
GSPO-N
1.696
3.712
4.943
Llama-3.2-3B-Instruct
3.937
5.333
6.062
GRPO
2.513
2.888
3.694
GRPO-N
2.633
3.042
4.370
Qwen2.5-Math-7B
0.118
-
0.134
GRPO
0.063
-
0.054
GRPO-N
0.124
-
0.092
Token-Level
GSPO
0.087
-
0.065
Entropy
GSPO-N
0.095
-
0.075
Llama-3.2-3B-Instruct
0.127
-
0.107
GRPO
0.123
-
0.089
GRPO-N
0.105
-
0.107
C.6
Full Procedure for Case Study
A comparison of the correct responses of GRPO and GRPO-N in mathematical reasoning task (a test case from AMC
2023). The base model is Qwen2.5-Math-7B, which is trained on the training set of MATH dataset.
GRPO: It exhibits a tendency to repeatedly make similar errors, such as consistently generating code with TypeError
issues, and struggles to correct them.
To solve
the
problem , we need
to
f i n d
the maximum value
of
the
imaginary
p a r t
of
the
complex number
\ ( z \ )
t h a t
s a t i s f i e s
the
e q u a t i o n
\ ( | 1
+ z + z ^ 2 | =
4 \ ) .
Let ’ s
break down the
problem
step −by−s t e p .
1 .
Express
\ ( z \ )
in
terms
of
i t s
r e a l
and
imaginary
p a r t s :
Let
\ ( z = x + yi \ ) ,
where
\ ( x \ )
and
\ ( y \ )
are
r e a l
numbers ,
and
\ ( i \ )
i s
the
imaginary
u n i t .
Then
\ ( z ^2 = ( x + yi )^2 = x^2 −y^2 + 2 xyi \ ) .
22


## Page 23

A preprint
2 .
S u b s t i t u t e
\ ( z \ )
and
\ ( z ^ 2 \ )
i n t o
the
given
e qu a ti o n :
\ [
| 1 + z + z ^ 2 | = | 1 + ( x + yi ) + ( x^2 −y^2 + 2 xyi ) |
=
| ( 1
+ x + x^2 −y ^2) + ( y + 2xy ) i |
= 4
\ ]
The magnitude
of a complex number
\ ( a + bi \ )
i s
given by
\ ( \ s q r t {a ^2 + b ^ 2 } \ ) .
So we have :
\ [
\ s q r t {(1 + x + x^2 −y ^2)^2 + ( y + 2xy )^2} = 4
\ ]
Squaring
both
sides , we get :
\ [
(1 + x + x^2 −y ^2)^2 + ( y + 2xy )^2 = 16
\ ]
3 .
Maximize
the
imaginary
p a r t
\ ( y \ ) :
We need
to
f i n d
the maximum value
of
\ ( y \ )
t h a t
s a t i s f i e s
the
above
e qu a ti o n .
Let ’ s
use
Python and SymPy to
solve
t h i s .
‘ ‘ ‘ python
import sympy as
sp
# Define
v a r i a b l e s
x ,
y = sp . symbols ( ’ x y ’ ,
r e a l =True )
# Define
the
eq u at i o n
e q u a t i o n = (1 + x + x**2 −y **2)**2 + ( y + 2*x*y )**2 −16
# Solve
the
eq u at i on
f o r
y in
terms
of x
s o l u t i o n s = sp . solve ( equation ,
y )
# Find
the maximum value
of y
max_y = max ( [ sp . s i m p l i f y ( s o l )
f o r
s o l
in
s o l u t i o n s ] )
# Simplify
the maximum value
of y
max_y_simplified = sp . s i m p l i f y ( max_y )
#
E x t r a c t m and n from
the
s i m p l i f i e d
form
of max_y
m,
n = sp . f r a c t i o n ( max_y_simplified **2)
# Since m and n are
r e l a t i v e l y
prime , we can
d i r e c t l y
sum them
r e s u l t
= m + n
p r i n t ( r e s u l t )
‘ ‘ ‘
‘ ‘ ‘ ou tput
TypeError :
cannot
determine
t r u t h
value
of
R e l a t i o n a l
‘ ‘ ‘
The
e r r o r
occurs
because SymPy ’ s
‘ solve ‘
f u n c t i o n
can
r e t u r n
symbolic
s o l u t i o n s
t h a t
i n c l u d e
i n e q u a l i t i e s
or
o t h e r
r e l a t i o n a l
e x p r e s s i o n s .
To handle
t h i s , we need
to
ensure
t h a t we are
only
c o n s i d e r i n g
r e a l
s o l u t i o n s
and
then
f i n d
the maximum value
of
\ ( y \ ) .
Let ’ s
r e f i n e
the
approach by
s o l v i n g
the
eq u at i on
n u m e r i c a l l y
and
then
f i n d i n g
the maximum value
of
\ ( y \ ) .
We can
use SymPy ’ s
‘ nsolve ‘
f u n c t i o n
to
f i n d
numerical
s o l u t i o n s
and
then
determine
the maximum value
of
\ ( y \ ) .
23


## Page 24

A preprint
Here ’ s
the
r e v i s e d
code :
‘ ‘ ‘ python
import sympy as
sp
# Define
v a r i a b l e s
x ,
y = sp . symbols ( ’ x y ’ ,
r e a l =True )
# Define
the
eq u at i o n
e q u a t i o n = (1 + x + x**2 −y **2)**2 + ( y + 2*x*y )**2 −16
# Find
c r i t i c a l
p o i n t s
by
s o l v i n g
the
e qu a ti o n
c r i t i c a l _ p o i n t s
= sp . solve ( equation ,
y )
# We need
to
f i n d
the maximum value
of y
max_y = None
f o r
p o i n t
in
c r i t i c a l _ p o i n t s :
# Check
i f
the
s o l u t i o n
i s
r e a l
i f
p o i n t . i s _ r e a l :
i f
max_y
i s
None or
p o i n t > max_y :
max_y = p o i n t
# Simplify
the maximum value
of y
max_y_simplified = sp . s i m p l i f y ( max_y )
#
E x t r a c t m and n from
the
s i m p l i f i e d
form
of max_y
m,
n = sp . f r a c t i o n ( max_y_simplified **2)
# Since m and n are
r e l a t i v e l y
prime , we can
d i r e c t l y
sum them
r e s u l t
= m + n
p r i n t ( r e s u l t )
‘ ‘ ‘
‘ ‘ ‘ ou tput
TypeError :
unsupported
operand
type ( s )
f o r
** or pow ( ) :
’NoneType ’
and
’ int ’
‘ ‘ ‘
The
e r r o r
i n d i c a t e s
t h a t
the
‘max_y ‘
v a r i a b l e
i s
‘None ‘ ,
which means
t h a t
the
s o l u t i o n s
r e t u r n e d
by
‘ sp . solve ‘
did
not
y i e l d
any
r e a l
values
f o r
\ ( y \ ) .
This
s u g g e s t s
t h a t we need
to
handle
the
s o l u t i o n s
more
c a r e f u l l y
and
ensure
t h a t we are
only
c o n s i d e r i n g
r e a l
s o l u t i o n s .
Let ’ s
use a
d i f f e r e n t
approach
to
f i n d
the maximum value
of
\ ( y \ ) .
We can
use
numerical
methods
to
f i n d
the maximum value
of
\ ( y \ )
by
e v a l u a t i n g
the
s o l u t i o n s
over
a
range
of
\ ( x \ )
values .
Here ’ s
the
r e v i s e d
code
using
numerical
methods :
‘ ‘ ‘ python
import sympy as
sp
import numpy as np
# Define
v a r i a b l e s
x ,
y = sp . symbols ( ’ x y ’ ,
r e a l =True )
# Define
the
eq u at i o n
e q u a t i o n = (1 + x + x**2 −y **2)**2 + ( y + 2*x*y )**2 −16
24


## Page 25

A preprint
# Function
to
f i n d
the maximum value
of y
f o r
a
given x
def
find_max_y ( ) :
max_y = None
f o r
x_val
in
np . l i n s p a c e ( −10 ,
10 ,
1000):
s o l u t i o n s = sp . solve ( e qu a ti o n . subs ( x ,
x_val ) ,
y )
f o r
s o l
in
s o l u t i o n s :
i f
s o l . i s _ r e a l :
i f
max_y
i s
None or
s o l > max_y :
max_y = s o l
r e t u r n
max_y
# Find
the maximum value
of y
max_y = find_max_y ( )
# Simplify
the maximum value
of y
max_y_simplified = sp . s i m p l i f y ( max_y )
#
E x t r a c t m and n from
the
s i m p l i f i e d
form
of max_y
m,
n = sp . f r a c t i o n ( max_y_simplified **2)
# Since m and n are
r e l a t i v e l y
prime , we can
d i r e c t l y
sum them
r e s u l t
= m + n
p r i n t ( r e s u l t )
‘ ‘ ‘
‘ ‘ ‘ ou tput
t i m e o u t _ d e c o r a t o r . t i m e o u t _ d e c o r a t o r . TimeoutError :
’ Timed Out ’
‘ ‘ ‘
The
numerical
approach
can be
c o m p u t a t i o n a l l y
i n t e n s i v e
and
may not
be
the
most
e f f i c i e n t
way to
solve
t h i s
problem .
Instead ,
l e t ’ s
use a more
a n a l y t i c a l
approach
to
f i n d
the maximum value
of
\ ( y \ ) .
We can
use
the
f a c t
t h a t
the
eq u at i on
\ ( | 1
+ z + z ^ 2 | = 4 \ )
r e p r e s e n t s
a
c e r t a i n
geometric
shape
in
the
complex
plane .
To f i n d
the maximum value
of
the
imaginary
p a r t
of
\ ( z \ ) ,
we can
use
the
p r o p e r t i e s
of
complex numbers and
t h e i r
magnitudes .
Let ’ s
r e w r i t e
the
e q ua t io n
in
a more manageable
form . We have :
\ [
| 1 + z + z ^ 2 | = 4
\ ]
This
can be
w r i t t e n
as :
\ [
| 1 + x + yi + ( x + yi ) ^ 2 | = 4
\ ]
S i m p l i f y i n g
the
e x p r e s s i o n
i n s i d e
the
magnitude , we get :
\ [
| 1 + x + yi + ( x^2 −y^2 + 2 xyi ) |
= 4
\ ]
\ [
| 1 + x + x^2 −y^2 + ( y + 2xy ) i |
= 4
\ ]
The magnitude
of a complex number
\ ( a + bi \ )
i s
given by
\ ( \ s q r t {a ^2 + b ^ 2 } \ ) .
So we have :
\ [
\ s q r t {(1 + x + x^2 −y ^2)^2 + ( y + 2xy )^2} = 4
\ ]
Squaring
both
sides , we get :
\ [
25


## Page 26

A preprint
(1 + x + x^2 −y ^2)^2 + ( y + 2xy )^2 = 16
\ ]
To f i n d
the maximum value
of
\ ( y \ ) , we can
use
c a l c u l u s
to
f i n d
the
c r i t i c a l
p o i n t s
of
t h i s
e q ua t io n .
However ,
a more
s t r a i g h t f o r w a r d
approach
i s
to
r e c o g n i z e
t h a t
the maximum value
of
\ ( y \ )
w i l l
occur
when the
imaginary
p a r t
of
\ ( z \ )
i s
maximized .
By a n a l y z i n g
the
equation , we can
see
t h a t
the maximum value
of
\ ( y \ )
occurs when the
r e a l
p a r t
of
the
e x p r e s s i o n
i n s i d e
the
magnitude
i s
minimized
and
the
imaginary
p a r t
i s
maximized .
This
happens when
\ ( x = −\ f r a c {1}{2}\) ,
which
s i m p l i f i e s
the
e qu a ti o n
to :
\ [
\ l e f t (1 −\ f r a c {1}{2} + \ l e f t ( −\ f r a c {1}{2}\ r i g h t )^2
−y ^2\ r i g h t )^2 + ( y + 2\ l e f t ( −\ f r a c {1}{2}\ r i g h t ) y )^2 = 16
\ ]
S i m p l i f y i n g
f u r t h e r , we get :
\ [
\ l e f t (1 −\ f r a c {1}{2} + \ f r a c {1}{4} −y ^2\ r i g h t )^2 + ( y −y )^2 = 16
\ ]
\ [
\ l e f t ( \ f r a c {3}{4} −y ^2\ r i g h t )^2 = 16
\ ]
Taking
the
square
r o o t
of
both
sides , we get :
\ [
\ l e f t | \ f r a c {3}{4} −y ^2\ r i g h t |
= 4
\ ]
This
gives
us two
cases :
1 .
\ ( \ f r a c {3}{4} −y^2 = 4 \ )
2 .
\ ( \ f r a c {3}{4} −y^2 =
−4\)
Solving
the
f i r s t
case :
\ [
\ f r a c {3}{4} −y^2 = 4
\ ]
\ [
y^2 = \ f r a c {3}{4} −4 = −\ f r a c {13}{4}
\ ]
This
i s
not
a
v a l i d
s o l u t i o n
s i n c e
\ ( y ^ 2 \ )
cannot
be
n e g a t i v e .
Solving
the
second
case :
\ [
\ f r a c {3}{4} −y^2 = −4
\ ]
\ [
y^2 = \ f r a c {3}{4} + 4 = \ f r a c {19}{4}
\ ]
So the maximum value
of
\ ( y \ )
i s :
\ [
y = \ s q r t {\ f r a c {19}{4}} = \ f r a c {\ s q r t {19}}{2}
\ ]
Thus ,
the maximum value
of
the
imaginary
p a r t
of
\ ( z \ )
i s
\ ( \ f r a c {\ s q r t { 1 9 } } { 2 } \ ) .
The
va lue s
\ (m\ )
and
\ ( n \ )
are
19 and
2 ,
r e s p e c t i v e l y ,
and
they
are
r e l a t i v e l y
prime .
Therefore ,
\ (m + n = 19 + 2 =
2 1 \ ) .
The
f i n a l
answer
i s :
\ [
\ boxed {21}
26


## Page 27

A preprint
\ ]
GRPO-N: It exhibits a reduction in erroneous code generation relative to GRPO.
To solve
the
problem , we need
to
f i n d
the maximum imaginary
p a r t
of
a
complex number
\ ( z \ )
t h a t
s a t i s f i e s
\ ( | 1
+ z + z ^ 2 | =
4 \ ) .
Let ’ s
break
down the
problem
i n t o
manageable
s t e p s :
1 .
Express
\ ( z \ )
in
terms
of
i t s
r e a l
and
imaginary
p a r t s :
Let
\ ( z = x + yi \ ) ,
where
\ ( x \ )
and
\ ( y \ )
are
r e a l
numbers .
2 .
S u b s t i t u t e
\ ( z \ )
i n t o
the
given
e qu a ti o n :
The
e q u a t i on
becomes
\ ( | 1
+ ( x + yi ) + ( x + yi ) ^ 2 | =
4 \ ) .
Simplify
the
e x p r e s s i o n
i n s i d e
the
a b s o l u t e
value :
\ [
1 + ( x + yi ) + ( x + yi )^2 = 1 + x + yi + x^2 −y^2 + 2 xyi
= (1 + x + x^2 −y ^2) + ( y + 2xy ) i
\ ]
The magnitude
of
t h i s
complex number
i s :
\ [
\ s q r t {(1 + x + x^2 −y ^2)^2 + ( y + 2xy )^2} = 4
\ ]
3 .
Square
both
s i d e s
to
remove
the
square
r o o t :
This
gives
us
the
e q ua t io n :
\ [
(1 + x + x^2 −y ^2)^2 + ( y + 2xy )^2 = 16
\ ]
4 .
Find
the maximum value
of
\ ( y \ ) :
We need
to
maximize
\ ( y \ )
s u b j e c t
to
the
above
eq u at i on .
We can
use
c a l c u l u s
or
numerical
methods
to
f i n d
the maximum value
of
\ ( y \ ) ,
but
f o r now ,
l e t ’ s
proceed
with
a
numerical
approach
using
Python
to
f i n d
the maximum value
of
\ ( y \ ) .
We’ l l
use a
g r i d
search
method
to
f i n d
the maximum value
of
\ ( y \ ) .
We’ l l
i t e r a t e
over
a
range
of
\ ( x \ )
and
\ ( y \ )
values
and
heck which
values
s a t i s f y
the
eq u at i on .
We’ l l
keep
t r a c k
of
the maximum
\ ( y \ )
value
found .
Let ’ s
w r i t e
the
Python
code
to
perform
t h i s
search .
‘ ‘ ‘ python
import numpy as np
# Define
the
f u n c t i o n
t h a t
r e p r e s e n t s
the
e qu a ti o n
def
e q u a t i o n ( x ,
y ) :
r e t u r n
(1 + x + x**2 −y **2)**2 + ( y + 2*x*y )**2 −16
#
I n i t i a l i z e
v a r i a b l e s
to
keep
t r a c k
of
the maximum y value
max_y = −np . i n f
max_x = 0
# Define
the
range
and
s t e p
s i z e
f o r
x and y
x_vals = np . l i n s p a c e ( −5 ,
5 ,
500)
y_vals = np . l i n s p a c e ( −5 ,
5 ,
500)
#
I t e r a t e
over
the
range
of x and y
values
f o r
x in
x_vals :
f o r
y in
y_vals :
i f
np . i s c l o s e ( e q ua t i on ( x ,
y ) ,
0 ,
a t o l =1e −5):
i f
y > max_y :
max_y = y
max_x = x
27


## Page 28

A preprint
p r i n t ( ( max_y ,
max_x ) )
‘ ‘ ‘
‘ ‘ ‘ ou tput
( −0.030060120240481286 ,
−1.997995991983968)
‘ ‘ ‘
The
numerical
method
s u g g e s t s
t h a t
the maximum value
of
\ ( y \ )
i s
approximately
\ ( −0 . 0 3 \ ) ,
which
i s
very
c l o s e
to
0.
This
i n d i c a t e s
t h a t
the maximum imaginary
p a r t
of
\ ( z \ )
i s
l i k e l y
0 ,
but
l e t ’ s
v e r i f y
t h i s
by checking
the
exact
form
of
the
s o l u t i o n .
To f i n d
the
exact maximum imaginary
part , we can
use
the
f a c t
t h a t
the
e q ua t io n
\ ( | 1
+ z + z ^ 2 | = 4 \ )
r e p r e s e n t s
a
curve
in
the
complex
plane .
We can
analyze
t h i s
curve
to
f i n d
the maximum imaginary
p a r t
a n a l y t i c a l l y .
Let ’ s
c o n s i d e r
the
e q ua t io n
again :
\ [
(1 + x + x^2 −y ^2)^2 + ( y + 2xy )^2 = 16
\ ]
To maximize
\ ( y \ ) , we can
look
a t
the
c r i t i c a l
p o i n t s
of
t h i s
e q u at i on .
We can
use
the
method
of
Lagrange
m u l t i p l i e r s
or
d i r e c t l y
analyze
the
eq u at i o n
to
f i n d
the maximum value
of
\ ( y \ ) .
By a n a l y z i n g
the
equation , we can
see
t h a t
the maximum imaginary
p a r t
occurs when
\ ( x =
−1 / 2 \ ) .
S u b s t i t u t i n g
\ ( x =
−1/2\)
i n t o
the
equation , we get :
\ [
(1 −\ f r a c {1}{2} + \ f r a c {1}{4} −y ^2)^2 + ( y −y )^2
= 16
\ i m p l i e s
\ l e f t ( \ f r a c {3}{4} −y ^2\ r i g h t )^2 = 16
\ ]
Solving
f o r
\ ( y \ ) , we get :
\ [
\ f r a c {3}{4} −y^2 = \pm 4
\ i m p l i e s
y^2 = \ f r a c {3}{4}
\pm 4
\ ]
Since
\ ( y ^ 2 \ )
must be non−negative , we take
the
p o s i t i v e
r o o t :
\ [
y^2 = \ f r a c {3}{4} + 4 = \ f r a c {19}{4}
\ i m p l i e s
y = \pm \ f r a c {\ s q r t {19}}{2}
\ ]
The maximum imaginary
p a r t
i s
\ ( \ f r a c {\ s q r t { 1 9 } } { 2 } \ ) .
Therefore ,
\ (m = 1 9 \ )
and
\ ( n = 2 \ ) ,
so
\ (m + n =
2 1 \ ) .
Thus ,
the
f i n a l
answer
i s :
\ [
\ boxed {21}
\ ]
28



---

# Paper 3: 2604.18381v1(base paper close)


**Source file:** `2604.18381v1(base paper close).pdf`

**Pages:** 12


## Page 1

LEARNING FROM LESS: MEASURING THE EFFECTIVENESS OF RLVR IN LOW
DATA AND COMPUTE REGIMES
Justin Bauer 1 Thomas Walshe 1 * Derek Pham 1 Harit Vishwakarma 1 †
Armin Parchami 1 Frederic Sala 1 2 Paroma Varma 1
ABSTRACT
Fine-tuning Large Language Models (LLMs) typically relies on large quantities of high-quality annotated data, or
questions with well-defined ground truth answers in the case of Reinforcement Learning with Verifiable Rewards
(RLVR). While previous work has explored the benefits to model reasoning capabilities by scaling both data and
compute used for RLVR, these results lack applicability in many real-world settings where annotated data and
accessible compute may be scarce. In this work, we present a comprehensive empirical study of open-source
Small Language Model (SLM) performance after RLVR in low data regimes. Across three novel datasets covering
number counting problems, graph reasoning, and spatial reasoning, we characterize how model performance scales
with dataset size, diversity, and complexity. We demonstrate that (1) procedural datasets allow for fine-grained
evaluation and training dataset development with controllable properties (size, diversity, and complexity), (2)
under RLVR, models trained on lower complexity tasks can generalize to higher complexity tasks, and (3) training
on mixed complexity datasets is associated with the greatest benefits in low data regimes, providing up to 5×
sample efficiency versus training on easy tasks. These findings inspire future work on the development of data
scaling laws for RLVR and the use of procedural data generators to further understand effective data development
for efficient LLM fine-tuning.
1
INTRODUCTION
Recent advances in Large Language Models (LLMs) have
achieved significant improvements in reasoning capabili-
ties (OpenAI, 2025a; Comanici et al., 2025; OpenAI, 2025b;
Zeng et al., 2025; Anthropic, 2025); this has, in part, been
driven by the adoption of Reinforcement Learning with
Verifiable Rewards (RLVR) (Shao et al., 2024; Wen et al.,
2025). RLVR provides an effective method for post-training
LLMs by rewarding models based on verifiable outcomes
(e.g., answers that can be compared to a known unambigu-
ously correct ground truth) rather than noisy human prefer-
ences (Poddar et al., 2024). For classes of problems with
verifiable outcomes, such as in mathematics (Wang et al.,
2025b), the adoption of RLVR has enabled models (e.g.,
DeepSeek R1 (Guo et al., 2025)) to achieve state-of-the-art
performance and allowed strong problem solving and self-
correction capabilities to emerge. However, many of these
advances are made under the assumption that high-quality
training data and compute are abundant (Khatri et al., 2025).
∗Work done at Snorkel AI. Now at Reflection AI. †Work
done at Snorkel AI. Now at University of Oxford. 1Snorkel AI
2University of Wisconsin-Madison. Correspondence to: Justin
Bauer <justin.bauer@snorkel.ai>.
Proceedings of the 9 th MLSys Conference, Bellevue, WA, USA,
2026. Copyright 2026 by the author(s).
Recent RLVR research has focused on using high vol-
umes of question-answer pairs to improve the reasoning
capabilities of LLMs. For example, DeepMath-103K in-
cludes over 100,000 challenging and decontaminated sam-
ples for training (He et al., 2025). However, in realistic,
resource-constrained situations, where both annotated data
and compute may be limited, these results may be chal-
lenging to replicate or extend to new reasoning domains.
While previous studies have explored scaling RLVR with
respect to model size and compute budget (Khatri et al.,
2025; Tan et al., 2025), or focused on reducing compute
requirements (both through Small Language Model (SLM)
fine-tuning (Dang & Ngo, 2025) and Low-Rank Adaptation
(LoRA) (Wang et al., 2025a)), there has been less atten-
tion on data scaling and the effectiveness of RLVR in low
data regimes. In this work, we focus on characterizing how
the size, diversity, and complexity of training data influ-
ence the reasoning capabilities of models. Motivated by
these ideas, we specifically investigate the research ques-
tion: “How does model performance evolve when training
data and compute are limited, and what characteristics of
data impact generalization in such regimes?”
We conduct a systematic empirical study using open-source
SLMs fine-tuned using RLVR under low data and compute
regimes to help understand these relationships, the results
arXiv:2604.18381v1  [cs.AI]  20 Apr 2026


## Page 2

Measuring the Effectiveness of RLVR in Low Data and Compute Regimes
of which help to shape data scaling laws that describe the
effectiveness of RLVR in different scenarios. For our ex-
periments, we introduce three novel datasets that allow data
to be generated procedurally with desired volume, diver-
sity, and complexity. These procedural datasets allow us
to better isolate the influence of different attributes (e.g.,
topic of questions, complexity of questions, etc.) and study
fine-tuning dynamics in controlled settings. We summarize
our contributions and findings as follows:
• New procedural datasets for reasoning tasks. We de-
velop three new datasets designed to support RLVR that
cover number counting problems, graph reasoning, and
spatial reasoning. For each of the datasets, we report
empirical solving rates across 10 models (including open-
source and proprietary LLMs), demonstrating the value
of using procedural data in creating challenging and com-
plex tasks.
• Study of RLVR in low data regimes. Using an open-
source LLM (Qwen3-4B), we investigate the relation-
ship between dataset size and composition (focusing on
task complexity) and performance after fine-tuning. Dif-
ferent training data configurations are used, capturing a
range of sizes and complexities across the three dataset
types. We observe that models trained on small volumes
of lower complexity tasks (i.e., easier questions) general-
ize to more complex tasks, and that training on a mixed
complexity dataset is associated with up to 5× the sample
efficiency under the same data budget.
2
RELATED WORK
2.1
Scaling Laws for Language Models
Early work by Kaplan et al. (2020) and Hoffmann et al.
(2022) established predictable relationships between model
performance, size, data, and compute, showing that increas-
ing training data and parameters yields smooth performance
improvements under fixed budgets. These analyses primar-
ily characterize how performance scales with model size and
compute, but do not address the effect of dataset composi-
tion, particularly difficulty distribution, under fixed budgets.
Zhang et al. (2024) extended these ideas to supervised fine-
tuning, showing that downstream loss depends jointly on
fine-tuning data size, model size, and parameter-efficient
adaptation methods.
2.2
Reinforcement Learning Scaling for LLMs
Recent work has explored how RL affects post-training per-
formance in LLMs. Khatri et al. (2025) present ScaleRL, a
large-scale framework characterizing RL performance under
different compute budgets and algorithmic choices, provid-
ing valuable insights into efficiency at scale. However, their
study primarily emphasizes compute scaling rather than
data composition or limited-data regimes. Tan et al. (2025)
analyze RL post-training in mathematical reasoning tasks
using Qwen2.5 models (0.5B–14B), finding that larger mod-
els achieve higher sample efficiency and that moderate data
reuse (≤25 epochs) approaches the performance of unique
data. However, their analysis is confined to a single domain
and does not explore dataset composition or difficulty ef-
fects. Lai et al. (2025) provide a comprehensive overview
of post-training methods, including supervised fine-tuning
and RL from feedback. Their taxonomy outlines general
scaling trends but remains largely descriptive, underscoring
the need for empirical studies focused on data composition
rather than model or compute scaling.
2.3
Data Efficiency and Selection in RL
Efficient data utilization has been a recurring challenge in
RL fine-tuning. Shen et al. (2025) identify two major bottle-
necks in RLHF data scaling—reward hacking and reduced
response diversity—and propose hybrid reward systems
and prompt-selection strategies that emphasize harder, low-
reward prompts to stabilize training and improve reasoning.
Their focus on selecting informative examples aligns with
our investigation into how dataset composition and diffi-
culty influence RL performance under limited data. Li et al.
(2025) introduce LIMR (Less is More for RL), showing
that small amounts of carefully curated data can outperform
larger datasets. By quantifying each sample’s contribution
to learning, they demonstrate that 1.4K selected samples
can match the performance of 8.5K unfiltered ones on math-
ematical reasoning tasks. While they provide a selection
heuristic, they do not analyze how varying dataset size or
difficulty affects performance, which our study examines.
2.4
Verification and Reasoning
Work on verifiable rewards has advanced understanding of
how structured feedback can drive reasoning improvements.
Wen et al. (2025) demonstrate that RLVR expands genuine
reasoning ability rather than simply improving sampling
efficiency, introducing CoT-Pass@K metrics to measure
both reasoning and answer correctness. Liu et al. (2025)
propose RISE, an online RLVR framework that jointly opti-
mizes problem-solving and self-verification, improving ver-
ification accuracy and test-time robustness. These insights
inform our reward design, which combines correctness veri-
fication with format and conciseness components to provide
denser feedback in low-data training settings.
Our study complements these lines of research by holding
model size and compute fixed and isolating the effect of
data composition on RLVR effectiveness, motivating future
work on budget-aware RLVR theory that captures interac-
tions between optimization budget, token limits, and reward
sparsity.


## Page 3

Measuring the Effectiveness of RLVR in Low Data and Compute Regimes
3
METHODOLOGY
3.1
Datasets
To evaluate model performance with controlled training data,
we designed three programmatically generated datasets: (1)
Counting Problems, (2) Graph Reasoning, and (3) Spatial
Reasoning. Each dataset was constructed using pre-defined
code templates that allow for parameterization and con-
trolled variation across a well-defined taxonomy of oper-
ators, ranges, and/or conditions. All data instances con-
tain verifiable outcome-level ground truth, enabling both
quantitative evaluation and use in RLVR training pipelines
without relying on costly annotation or less reliable verifica-
tion methods (e.g., LLM-as-judge evaluation). The dataset
taxonomies are hand-crafted to promote diversity through
meaningful variation rather than to provide statistical guar-
antees of coverage. Across all three datasets, the procedural
generators control multiple correlated instance properties
simultaneously (e.g., graph size and edge density, opera-
tor families and step counts, number of actions and query
types), so comparisons between Easy and Mixed training
configurations reflect variation along several dimensions
rather than a single axis of complexity.
3.1.1
Counting Problems Dataset
The Counting Problems dataset is a procedurally generated
benchmark designed to evaluate the numerical reasoning
and pattern recognition capabilities of language models in
constrained computational tasks. The dataset employs a tem-
plating methodology to generate questions with determin-
istic programmatic ground-truth answers, enabling precise
control over problem complexity and systematic evaluation
of model performance. Each question is composed of:
• A natural language prompt that specifies a counting task
over a sequence of integers within a defined range.
• A sequence of conditional filters and transformations to
apply before the final counting operation.
• A deterministic ground-truth answer computed by pro-
grammatically executing the specified operations.
This design allows for direct assessment of multi-step rea-
soning and numerical manipulation, as models must accu-
rately parse operation sequences, track intermediate states,
and perform multi-step reasoning to produce correct an-
swers.
Controllable Complexity. We control problem complexity
through multiple structural dimensions. Primarily, we vary
the range scale, which controls the magnitude of integer
ranges from which values are drawn. We also manipulate
the operator diversity through a taxonomy of counting and
aggregation operations:
• Basic Counting: Count, Unique Count, Zero Count.
• Conditional Counting: Even Count, Odd Count, Posi-
tive Count, Negative Count, Divisible By N Count.
• Threshold-Based:
Below Threshold Count, Above
Threshold Count.
• Arithmetic Aggregation: Sum, Product, Mean, Median,
Mode.
• Extrema Operations: Min, Max, Range.
• Bitwise Operations: Bitwise AND, Bitwise OR, Bitwise
XOR, Bitwise NAND.
Additionally, we vary compositional depth through the num-
ber of conditional filters (1–4) and transformations (0–3)
applied before the final operation, creating problems with
1–7 total intermediate steps. This yields a spectrum from
simple counting to complex multi-step compositional rea-
soning.
Example. The following is an example counting problem.
Question: Consider the integers from 1 to 100, inclusive.
First, keep only the numbers that are even. Then, keep only
the numbers that are divisible by 3. Of these numbers, count
how many values remain.
Solution: 16 (computed programmatically by executing:
|{x ∈[1, 100] : x mod 2 = 0 ∧x mod 3 = 0}| = 16)
Evaluation Protocol. Each model receives one prompt per
question and generates a completion containing reasoning
(optional) and a final numerical answer. Numeric responses
are parsed using regular expressions and programmatically
validated via exact-match comparison against the determin-
istic ground truth.
3.1.2
Graph Reasoning Dataset
The Graph Reasoning dataset is a procedurally generated
benchmark designed to evaluate the mathematical and spa-
tial reasoning capabilities of language models over graph-
structured problems. The dataset extends the templating
methodology to formal graph-based domains, allowing pre-
cise control over question complexity and solution verifia-
bility. Each question is composed of:
• A natural language operator that defines a computation
over a graph (e.g., “Find the minimum vertex cover of an
undirected graph”).
• A graph structure encoded textually as lists of nodes and
edges.
• A verifiable ground-truth solution, computed via deter-
ministic algorithms.
This design allows for direct assessment of multi-hop rea-
soning and long-context tracking, as models must parse
graph representations and perform symbolic reasoning to
produce correct answers.
Controllable Complexity. We control problem complexity
through several factors. Primarily, we vary the graph size


## Page 4

Measuring the Effectiveness of RLVR in Low Data and Compute Regimes
(number of nodes and edges), ranging from small (5 nodes)
to large (25 nodes) graphs, which increases the reasoning
load and relational tracking required. We also vary the
operator diversity, drawing from a predefined taxonomy
of graph-theoretic operations that span multiple problem
families:
• Subgraph Optimization: Minimum Density Subgraph,
Maximum Clique, Maximum Independent Set, Minimum
Vertex Cover, Maximum Induced Bipartite Subgraph,
Acyclic Subgraph, Dense Subgraph Variants.
• Graph Partitioning: Balanced Cut.
• Feedback Set Problems: Feedback Vertex Set, Feedback
Edge Set.
• Path Problems: Longest Path, Hamiltonian Path, Hamil-
tonian Cycle.
• Graph Metrics: Graph Diameter, Graph Radius, Graph
Density.
Additionally, some instances include weighted and directed
edges to introduce further structural variation, though these
are not treated as primary experimental variables.
Example. The following is an example question.
Question: Find the maximum independent set of an undi-
rected graph with 5 nodes. Find the largest set of vertices
with no edges between them. If multiple maximum indepen-
dent sets exist, return any one of them.
Graph: Nodes: [0, 1, 2, 3, 4]; Edges: [(0,2), (0,4)].
Solution: The maximum independent set is [1, 2, 3, 4].
Evaluation Protocol. Each model receives one prompt
per question and is required to generate both a complete
reasoning trace and a final answer. The responses are subse-
quently normalized by a secondary model (GPT-4o) into a
canonical internal representation. A programmatic validator,
implemented using graph-based libraries (e.g., networkx),
then verifies each output for correctness against the ground
truth or determines whether it constitutes a valid equivalent
solution.
3.1.3
Spatial Reasoning Dataset
The Spatial Reasoning dataset evaluates spatial reasoning
capabilities of language models. We generate the problems
with varying difficulty level following the spatial reasoning
setup introduced in (Dsouza et al., 2025). Similar to the
graph reasoning and counting problems, this setting extends
the templating methodology and allows precise control over
question complexity and solution verifiability. Each ques-
tion is composed of:
• A description of a 2D spatial reasoning environment con-
sisting of a square grid (board) and a set of particles on the
board. The board and particles are located in a 2D space
and are oriented towards one of the cardinal directions
(East, North, West, South).
• A sequence of movement and rotation actions applied to
the board and particles.
• A query about the absolute or relative location or orien-
tation of the entities (board or particles) after all actions
have been applied.
• A verifiable ground-truth solution, obtained by program-
matically executing the simulation.
These problems test LLMs’ abilities to track and reason over
the location and orientation of entities in 2D space. The
division of problems between absolute and relative is moti-
vated by the fundamental dichotomy of egocentric (relative)
and allocentric (absolute) spatial reasoning depending on
the frame of reference (Denis, 2017).
Controllable Complexity. We control the complexity via
the number of actions and the type of query. Intuitively,
the problems with more actions and the ones based on the
relative spatial reasoning are expected to be more complex.
Example. The following is a sample problem.
Question: Consider a square grid of size 20×20 centered at
(0, 0). It has two particles P1 and P2 at locations (−1.5, 2.5)
and (3.5, 1.5), respectively. P1 and P2 face towards East
and West, respectively. P1 moves 1 step forward and P2
moves 1 step backwards. What is the location of P1, relative
to P2?
Solution: The location of P1 relative to P2 is (−5.0, 1.0).
Evaluation Protocol. Models are prompted with the ques-
tion and asked to generate a completion containing an op-
tional explanation and answer in structured (JSON) format.
If the original response has parsing errors, we fall back
to parsing with a secondary model (GPT-4o). The struc-
tured response is then compared against the ground truth.
For floating-point numbers, we match to the first 3 decimal
places and perform an exact match for integer and string
values.
3.2
Curation
For our model training experiments, we generated over
1,500 programmatically defined problems for each dataset
described above. Following dataset generation, we con-
ducted model-based evaluation runs based on each dataset’s
evaluation protocol across 10 diverse foundation models
spanning several model families (GPT, Claude, Gemini,
Grok, Llama, and Qwen).
Each model was evaluated with a single inference call per
data point, and aggregate pass rates were computed at the
instance level. Figure 1 shows each dataset’s performance
across the ten LLMs we used. To enable finer-grained diffi-
culty control, we categorized all problems into three diffi-
culty tiers based on the percentage of models that answered


## Page 5

Measuring the Effectiveness of RLVR in Low Data and Compute Regimes
gpt-5
gpt-4.1
gpt-oss-120b
claude-sonnet-4
claude-3-5-haiku
gemini-2.5-flash
grok-3
Llama-4-Maverick
Qwen3-1.7B
Qwen3-4B
Model
0
25
50
75
100
Accuracy (%)
Model-based evaluation results across datasets
Counting Problems
Graph Reasoning
Spatial Reasoning
Figure 1. Overall model-based evaluation results across all gen-
erated samples for Counting Problems, Graph Reasoning, and
Spatial Reasoning.
correctly:
• Easy: 67–100% of models answered correctly.
• Medium: 34–66% of models answered correctly.
• Hard: 0–33% of models answered correctly.
We then curated multiple subsets for downstream use by
sampling based on difficulty. The following dataset configu-
rations were curated:
• Easy training subsets: 100, 200, and 500 examples
sampled from the Easy tier.
• Mixed training subsets: 100, 200, and 500 examples
drawn across Easy, Medium, and Hard tiers (approxi-
mately 33% each).
• Test subset: 200 examples (500 examples for Graph Rea-
soning), not used in training, drawn across Easy, Medium,
and Hard tiers.
This multi-model calibration and stratified curation distin-
guishes difficulty based on performance across a diverse
range of models, ensuring that difficulty labels reflect ac-
tual model capabilities across architectures rather than hu-
man assumptions. Because multiple instance properties co-
vary with difficulty tier, observed performance differences
between Easy and Mixed configurations should not be at-
tributed to a single factor. The result is a set of practical
datasets that capture realistic capability boundaries across
the current LLM frontier, enabling controlled studies of how
problem difficulty interacts with fine-tuning dataset scale.
All training, validation, and test splits are strictly disjoint to
prevent data contamination or leakage between stages.
3.3
Fine-tuning
We employ RLVR to fine-tune SLMs on each problem do-
main. This approach enables direct optimization of task-
specific reward signals rather than relying on supervised
demonstrations, allowing the model to learn through explo-
ration and self-correction.
Base Model and Architecture. We use Qwen3-4B (Yang
et al., 2025) as our base model across all experiments, a
4-billion parameter model with strong reasoning capabili-
ties. To enable efficient fine-tuning on consumer hardware,
we apply LoRA (Hu et al., 2022) with rank r = 64 and
α = 16, targeting all linear layers. Recent work shows
LoRA matches full fine-tuning performance for reinforce-
ment learning even at low ranks (Schulman & Lab, 2025),
supporting its use in our compute-constrained setting. This
reduces trainable parameters to ∼100M while preserving
model expressiveness.
Training Algorithm. We use Group Relative Policy Op-
timization (GRPO) (Shao et al., 2024), a batch-wise ad-
vantage estimation algorithm designed for mathematical
reasoning. For each training example, we generate diverse
completions and compute advantages by comparing rewards
within each group. This approach reduces variance com-
pared to single-sample methods while maintaining explo-
ration. We optimize using AdamW with gradient clipping
(max norm 1.0) and apply a cosine learning rate schedule
with 10% warmup.
Reward Functions. We design task-specific reward func-
tions that balance correctness, reasoning quality, and re-
sponse format. We tried several formulations and report
the ones that performed best so far, though they are not
necessarily optimal and warrant further exploration.
Counting Rewards.
Multi-component reward (r
∈
[−0.4, +1.1]) combining binary correctness (r = 1.0 cor-
rect, r = 0.0 incorrect), format quality bonuses (+0.1 for
“Answer: X” format, +0.05 for acceptable variants, down to
−0.1 for invalid format), and reasoning step penalties (−0.1
per step beyond 5 steps, capped at −0.3). These compo-
nents apply to both correct and incorrect answers, creating
positive rewards for well-formatted responses and negative
rewards (r ∈[−0.4, 0]) for verbose incorrect answers.
Graph Reasoning Rewards.
Structured reward (r ∈
Table 1. Reinforcement learning hyperparameters across datasets.
Hyperparameter
Counting
Graph Reasoning
Spatial Reasoning
Shared Architecture Parameters
Base Model
Qwen3-4B
LoRA Rank
64
LoRA Alpha
16
Training Parameters
Training Steps
300
300
1000
Learning Rate
5 × 10−5
5 × 10−5
5 × 10−5
Batch Size (per GPU)
2
1
1
Num GPUs
4
4
4
Effective Batch Size
8
4
4
Generation Parameters
Generations per Prompt (K)
8
8
5
Temperature (τ)
1.0
1.0
1.0
Max Prompt Length
4096
4096
4096
Max Completion Length
2048
2048
2048
Optimization Parameters (Shared)
Optimizer
AdamW
Gradient Clipping
1.0
LR Schedule
Cosine with 10% warmup
Evaluation Frequency
Every 50 steps


## Page 6

Measuring the Effectiveness of RLVR in Low Data and Compute Regimes
[−0.2, +1.1]) combining binary correctness (r = 1.0 cor-
rect, r = 0.0 incorrect) and format quality bonuses (+0.1
for proper {"answer":
...} JSON format). Incorrect
but well-formatted responses receive partial credit (r = 0.1),
while unstructured or excessively long outputs incur penal-
ties (r = −0.2). This design reuses benchmark validation
for consistent correctness evaluation.
Spatial Reasoning Rewards. Binary exact-match reward
(r ∈{0, 1}) using query-specific validation methods. An-
swer extraction supports flexible JSON formatting in multi-
ple patterns (direct objects, code blocks) to accommodate
diverse model response strategies.
Training Configuration. Table 1 summarizes our training
hyperparameters. All models are trained on 4× NVIDIA
A100 80GB GPUs, with training times ranging from 5–12
hours depending on problem complexity and dataset size.
Evaluation Protocol. During training, we monitor valida-
tion performance every 50 steps on a held-out 10% split
from the training distribution. After training is complete,
we evaluate on the curated test subset (Section 3.2). We
use greedy decoding (temperature 0) for test evaluation to
assess the model’s most confident predictions. Test accu-
racy serves as our primary metric for comparing scaling
behaviors across dataset sizes and difficulty distributions.
4
RESULTS
We performed a range of RL fine-tuning experiments with
the curated datasets and training setup outlined in the pre-
vious section. Our goal is to empirically compare data
curation strategies under fixed training constraints, rather
than to isolate a single causal mechanism. In this section,
we first break down the training results on a per-dataset
basis, then discuss the high-level implications of fine-tuning
on programmatically generated data. Aggregate test accura-
cies are reported in Table 2; per-difficulty breakdowns are
provided alongside each dataset’s training curves.
4.1
Dataset Results
4.1.1
Counting Problems
Training-Time Observations. Figure 2a shows the training
progression for both easy-difficulty and mixed-difficulty.
Under a fixed 300-step training budget, models trained
on mixed-difficulty data overall showed continued valida-
tion improvement through step 300, suggesting that larger
datasets may require proportionally more training steps—
under compute constraints, small diverse datasets outper-
form large homogeneous ones.
Easy-difficulty training showed greater variation across
dataset sizes. The 100-example model exhibited severe
instability, peaking at 0.89 validation reward (step 150) be-
fore declining to 0.59 (step 300), coinciding with gradient
norm spikes exceeding 850× baseline values (Figure 4).
The 200- and 500-example models trained stably (final re-
wards 0.90 and 0.68), suggesting a minimum dataset size
threshold between 100 and 200 examples for stable opti-
mization. Notably, Mixed-100 trained stably despite the
same sample count, indicating that difficulty diversity can
substitute for dataset size in stabilizing RLVR training under
limited data and compute. To better understand what drives
aggregate reward variation, we decompose completions into
correctness and format categories (Figure 5). For counting,
the breakdown confirms that the Easy-100 collapse is driven
by a drop in correctness rate rather than format or extraction
issues.
Test-Time Generalization. Test performance confirmed
consistent improvement for the easy models, and showed a
degradation in performance for mixed models. Mixed mod-
els achieved 0.478 (100 examples), 0.476 (200 examples),
and 0.367 (500 examples) mean reward, with corresponding
solving accuracies of 50.0%, 50.5%, and 40.0%. Easy-
trained models scaled monotonically from 0.218 to 0.461,
but required 5× more examples to match mixed baselines
(500 easy examples ≈100 mixed examples in final accu-
racy).
Figure 3a decomposes test performance by question diffi-
culty. Mixed-trained models show performance degradation
with scaling. In contrast, easy-trained models scale mono-
tonically. The 100-example mixed model maintains the most
balanced cross-difficulty profile, while easy-trained models
require 500 examples to match mixed-trained performance
on easy questions.
Results demonstrate two novel scaling behaviors: (1) De-
graded scaling under compute constraints—however, with
continued validation improvement toward the final step, we
hypothesize that with more compute, we could see poten-
tial inverted-U scaling, contradicting supervised fine-tuning
scaling laws that predict monotonic improvement (Zhang
et al., 2024); and (2) 5× sample efficiency of diverse train-
ing data, suggesting that data composition may outweigh
data quantity in low-resource RL regimes. These findings
suggest that practitioners facing compute constraints should
prioritize dataset diversity over size.
4.1.2
Graph Reasoning
Training-Time Observations. Graph reasoning showed
stable but modest improvement in validation performance
across all training runs, except for the Mixed-100 configu-
ration (Figure 2b). In contrast, all easy-only runs showed
mostly consistent upward reward trajectories through step
300, indicating continued learning despite high training re-
ward variance. As shown in Table 2, the Easy-500 model
achieved the strongest test performance overall, suggesting


## Page 7

Measuring the Effectiveness of RLVR in Low Data and Compute Regimes
0
50
100
150
200
250
300
Training Step
0.4
0.2
0.0
0.2
0.4
0.6
0.8
1.0
1.2
Reward
Training & Validation Rewards - Easy Training Data
100 examples
200 examples
500 examples
0
50
100
150
200
250
300
Training Step
0.4
0.2
0.0
0.2
0.4
0.6
0.8
1.0
1.2
Reward
Training & Validation Rewards - Mixed Training Data
100 examples
200 examples
500 examples
(a) Counting: training and validation reward over 300 steps. Note the instability in the easy 100-example model (collapse after step 200).
50
100
150
200
250
300
Training Step
0.2
0.0
0.2
0.4
0.6
0.8
1.0
1.2
Reward
Training & Validation Rewards - Easy Training Data
100 examples
200 examples
500 examples
50
100
150
200
250
300
Training Step
0.2
0.0
0.2
0.4
0.6
0.8
1.0
1.2
Reward
Training & Validation Rewards - Mixed Training Data
100 examples
200 examples
500 examples
(b) Graph: training and validation reward over 300 steps. Easy-only training (left) achieves positive validation rewards; mixed-difficulty
(right) shows consistently negative rewards due to incomplete rollouts under token constraints.
0
200
400
600
800
1000
Training Step
0.0
0.2
0.4
0.6
0.8
1.0
Reward
Training & Validation Rewards - Easy Training Data
100 examples
200 examples
500 examples
0
200
400
600
800
1000
Training Step
0.0
0.2
0.4
0.6
0.8
1.0
Reward
Training & Validation Rewards - Mixed Training Data
100 examples
200 examples
500 examples
(c) Spatial: training and validation reward over 1000 steps. Binary reward (r ∈{0, 1}) creates discrete performance levels. Both regimes
show steady improvement.
Figure 2. Training reward curves across all three datasets (left: easy-only, right: mixed-difficulty). Colors: blue = 100, orange = 200,
green = 500 examples. Light shaded lines show training rewards; dark solid lines with diamond markers show validation rewards.


## Page 8

Measuring the Effectiveness of RLVR in Low Data and Compute Regimes
Easy
Medium
Hard
Question Difficulty
0.0
0.2
0.4
0.6
0.8
1.0
1.2
Mean Reward
0.530
(n=44)
0.187
(n=73)
0.052
(n=83)
0.888
(n=44)
0.487
(n=73)
0.084
(n=83)
0.931
(n=44)
0.556
(n=73)
0.134
(n=83)
Mean Reward
100 examples
200 examples
500 examples
Test Performance by Difficulty - Easy Training Data
Easy
Medium
Hard
Question Difficulty
0.0
0.2
0.4
0.6
0.8
1.0
1.2
Mean Reward
0.934
(n=44)
0.589
(n=73)
0.125
(n=83)
0.922
(n=44)
0.564
(n=73)
0.130
(n=83)
0.795
(n=44)
0.439
(n=73)
0.053
(n=83)
Mean Reward
100 examples
200 examples
500 examples
Test Performance by Difficulty - Mixed Training Data
(a) Counting: easy-trained models (left) specialize on easy questions but fail on harder ones; mixed-trained models (right) maintain
consistent cross-difficulty performance.
Easy
Medium
Hard
Question Difficulty
0.2
0.0
0.2
0.4
0.6
0.8
1.0
1.2
Mean Reward
0.671
(n=242)
0.214
(n=104)
-0.118
(n=154)
0.671
(n=242)
0.217
(n=104)
-0.133
(n=154)
0.710
(n=242)
0.259
(n=104)
-0.107
(n=154)
Mean Reward
100 examples
200 examples
500 examples
Test Performance by Difficulty - Easy Training Data
Easy
Medium
Hard
Question Difficulty
0.2
0.0
0.2
0.4
0.6
0.8
1.0
1.2
Mean Reward
0.618
(n=242)
0.169
(n=104)
-0.142
(n=154)
0.650
(n=242)
0.220
(n=104)
-0.109
(n=154)
0.679
(n=242)
0.209
(n=104)
-0.106
(n=154)
Mean Reward
100 examples
200 examples
500 examples
Test Performance by Difficulty - Mixed Training Data
(b) Graph: both training regimes achieve positive rewards on easy questions but struggle with medium and hard problems due to token
constraints.
Easy
Medium
Hard
Question Difficulty
0.0
0.2
0.4
0.6
0.8
1.0
Mean Reward
0.88
(n=68)
0.47
(n=66)
0.13
(n=66)
0.90
(n=68)
0.57
(n=66)
0.23
(n=66)
0.86
(n=68)
0.52
(n=66)
0.19
(n=66)
Mean Reward
100 examples
200 examples
500 examples
Test Performance by Difficulty - Easy Training Data
Easy
Medium
Hard
Question Difficulty
0.0
0.2
0.4
0.6
0.8
1.0
Mean Reward
0.89
(n=68)
0.55
(n=66)
0.25
(n=66)
0.88
(n=68)
0.51
(n=66)
0.23
(n=66)
0.85
(n=68)
0.56
(n=66)
0.25
(n=66)
Mean Reward
100 examples
200 examples
500 examples
Test Performance by Difficulty - Mixed Training Data
(c) Spatial: discrete binary rewards yield modest cross-distribution differences, with both models generalizing similarly.
Figure 3. Test accuracy by question difficulty across all three datasets (left: easy-trained, right: mixed-trained). Colors: blue = 100, orange
= 200, green = 500 examples. Bars show accuracy on the held-out test set.


## Page 9

Measuring the Effectiveness of RLVR in Low Data and Compute Regimes
Samples
Counting
Graph
Spatial
Easy
Mixed
Easy
Mixed
Easy
Mixed
0
31.3
31.3
29.4
29.4
26.1
26.1
100
21.9
44.2
33.3
29.1
49.9
56.6
200
40.0
43.4
32.9
32.7
56.7
54.3
500
44.2
35.5
36.5
34.0
53.1
55.7
Table 2. Mean test accuracy (%) vs training sample size across
all datasets and difficulty settings. First row shows base model
performance (Qwen3-4B with no fine-tuning).
that reinforcement learning was more stable and effective
on larger, easier datasets in this setting.
Mixed-difficulty training was more challenging. Validation
rewards were frequently negative across all dataset scales
(Figure 2b), reflecting incomplete rollouts where the model
exhausted its token budget before producing a parseable
structured output. On average, mixed datasets contained
larger graphs (14.9 nodes vs. 12.6 for easy sets), resulting in
longer input and output sequences that more often exceeded
the maximum generation length. Since the reward function
penalized cutoffs (Section 3.3), these incomplete rollouts
suppressed training rewards and slowed convergence. The
reward component breakdown (Figure 5) confirms that ex-
traction failures account for the majority of completions
across all graph configurations (59–65% for easy, 67–73%
for mixed), explaining the persistently negative aggregate
rewards. Notably, the Mixed-100 model performed slightly
worse than the baseline (29.1% vs. 29.4%), coinciding with
limited diversity and frequent negative reward signals from
incomplete rollouts.
Test-Time Generalization. Easy-trained models slightly
outperformed mixed-trained models overall at test time (Fig-
ure 3b). However, both training setups failed to generalize
beyond the easy test subset. Performance declined sharply
on medium problems and became mostly negative on hard
problems, indicating that the models struggled to reason
0
50
100
150
200
250
300
Training Step
10
2
10
1
100
101
Gradient Norm
Easy-100
Easy-200
Mixed-100
Figure 4. Gradient norm over training steps for three counting
configurations. Easy-100 exhibits spikes exceeding 850× baseline
between steps 150–300, coinciding with the reward collapse in
Figure 2a. Easy-200 and Mixed-100 remain stable throughout,
supporting a minimum diversity threshold for stable optimization.
Counting Problems
0
50
100
% of Completions
Easy-100
Easy-200
Easy-500
0
100
200
300
Training Step
0
50
100
% of Completions
Mixed-100
0
100
200
300
Training Step
Mixed-200
0
100
200
300
Training Step
Mixed-500
Correct + Format
Correct, No Bonus
Wrong, Extracted
Extraction Failed
Graph Reasoning
0
50
100
% of Completions
Easy-100
Easy-200
Easy-500
0
100
200
300
Training Step
0
50
100
% of Completions
Mixed-100
0
100
200
300
Training Step
Mixed-200
0
100
200
300
Training Step
Mixed-500
Correct + Format
Correct, No Bonus
Wrong, Extracted
Extraction Failed
Figure 5. Reward component breakdown across training configura-
tions. For counting, Easy-100 correctness collapses after step 150,
consistent with gradient norm instability (Figure 4). For graph
reasoning, extraction failures dominate (59–65% easy, 67–73%
mixed), explaining persistently negative aggregate rewards. Spatial
reasoning is excluded (binary reward, no sub-components).
over longer or more complex graphs. This is consistent
with training inefficiencies under compute and context con-
straints, which limited the model’s ability to complete rea-
soning and reduced opportunities to receive positive reward
signals on more difficult rollouts.
Scaling analysis across 100–500 examples revealed only
incremental improvements within each training regime (Fig-
ure 2b). Easy-trained models scaled roughly linearly in
final validation reward, while mixed-trained models showed
minimal change. This weak scaling trend supports the con-
clusion that training on harder instances—which involve
longer sequences and more reasoning steps—is constrained
by token limits and fixed compute budgets.
Overall, graph reasoning scaling was constrained more by
token generation limits than by data volume, given the ver-
bose nature of the problems. Larger graphs inflated input
and output lengths, causing the model to more frequently
exhaust its token budget before producing complete out-
puts. Increasing these limits would change the underlying
training dynamics and may yield different scaling behavior,
making length-adaptive optimization an important direction
for future work.


## Page 10

Measuring the Effectiveness of RLVR in Low Data and Compute Regimes
AL
AO
RL
RO
Query Type
33.33
58.60
7.55
12.50
Base
100
200
500
Training Samples
60.42
67.50
66.25
67.91
73.49
73.02
32.08
33.21
35.85
43.93
56.79
42.86
Easy Setting
100
200
500
Training Samples
58.33
61.25
65.00
72.09
70.23
68.37
30.94
30.19
36.98
67.50
58.93
55.71
Mixed Setting
0
20
40
60
80
100
Accuracy (%)
Figure 6. Test accuracy on different types of queries in spatial rea-
soning. Here, AO, AL stand for absolute orientation and absolute
location, and similarly, RO, RL stand for relative queries. In both
easy and mixed settings, we see improvements across all types of
queries. The improvements are more pronounced on location and
relative orientation queries. Moreover, in the mixed setting, the
performance is much better on the relative orientation queries.
4.1.3
Spatial Reasoning
Training-Time Observations. During training, we see that
the validation accuracy improves gradually in all settings
(Figure 2c). The rate of improvement varies across the
settings—the model learns faster in easy settings with fewer
samples in comparison to other settings with more samples,
especially with medium and hard difficulty levels. This is
expected behavior with a fixed training budget across these
settings; medium and hard questions are expected to be
harder to learn and may require a longer training time and
larger token limits.
Test-Time Generalization. After fine-tuning the model, we
evaluate it on a test set of 200 problems of mixed difficulty
levels (68 easy, 66 medium, and 66 hard) spanning 4 query
types: absolute location (48 questions), absolute orientation
(43), relative location (53), and relative orientation (56). We
draw the following insights from the results on the test data.
Fine-tuning with either type of training set improves accu-
racy. While this is expected from fine-tuning in general, as
a first step, it is necessary to study whether fine-tuning is
effective in this novel setting of spatial reasoning. These
results (in Table 2) provide clear evidence that fine-tuning
improves the model’s performance (up to 2×) on the given
spatial reasoning task.
Diminishing returns with more training data under compute
constraints. Contrary to conventional wisdom, more train-
ing data may not yield higher accuracy when the compute
budget for training is fixed. Our results on spatial reasoning
in both easy and mixed training settings in Table 2 support
this claim. We see that in the easy setting the accuracy im-
proves by about 7% when moving from 100 to 200 training
samples, but drops by 3.6% in the 500 samples case. In the
mixed setting, the accuracy does not change much with the
number of samples, but we see a small dip when compared
to the 100 samples setting. These results suggest an inverted
U-shape scaling of test accuracy with the training set size
under compute constraints.
Training on mixed difficulty samples generally performs
better in comparison to training only on easy samples. In
Table 2, we observe that training on samples with mixed
difficulty results in a higher or comparable test accuracy, in
contrast to training on an easy set of the same size. More-
over, the accuracy with 100 mixed samples even surpasses
the accuracy with 500 easy samples, suggesting training on
fewer samples of varying difficulty is preferable to training
on a large set of easy samples.
Improvements across all types of queries. Next, we study the
performance of the model on different types of queries be-
fore and after fine-tuning. The results are shown in Figure 6.
We observe that fine-tuning with either training set improves
accuracy across all query types. The baseline results suggest
that relative queries are harder, and it is interesting to see
that even training on easy samples increases the accuracy
on these queries significantly. Similar improvements are
also noted in the mixed setting, which are more pronounced
on the relative orientation queries. This may reflect the fact
that the mixed set contains more such queries in comparison
to the easy set.
Robustness considerations. Due to computational cost, we
did not perform multi-seed repetitions. We instead rely on
consistency of qualitative trends across 18 configurations (3
domains × 6 data configurations) as a robustness signal, sug-
gesting the observed effects are not solely driven by training
noise. The mixed-vs-easy sample efficiency trend emerges
independently in both Counting and Spatial Reasoning de-
spite differing reward structures, and all configurations share
fixed training budgets, token limits, and task-specific reward
functions, so performance differences reflect how dataset
size and composition interact with these constraints. We
view these findings as empirical observations under low data
and compute regimes that generate hypotheses rather than
definitive causal claims.
4.2
Implications
Our results across the datasets point to a unifying theme:
under fixed-budget RLVR regimes, performance is shaped
by interactions between dataset size, composition, training
duration, and token limits. We distill three design lessons
for practitioners.
1. Training-set composition can outweigh data volume.
In both Counting and Spatial Reasoning, small mixed-
difficulty datasets matched or exceeded the test accuracy of
larger easy-only datasets. For Counting, 100 mixed exam-
ples matched 500 easy examples in test accuracy, a 5× sam-
ple efficiency advantage, suggesting that curating across


## Page 11

Measuring the Effectiveness of RLVR in Low Data and Compute Regimes
empirically defined difficulty tiers can be more effective
than scaling up easy examples alone.
2. Under fixed training budgets, scaling data alone can
be ineffective. Larger datasets receive fewer optimization
updates per example under a fixed step budget. In Counting,
mixed-difficulty test accuracy declined beyond 100 exam-
ples even though validation rewards were still improving at
the final step. In Spatial Reasoning, easy-trained accuracy
peaked at 200 examples before declining, and mixed-trained
accuracy did not improve beyond 100 examples. These pat-
terns motivate joint consideration of dataset size, training
duration, and token budgets.
3. Harder instances can be constrained by incomplete
rollouts under fixed token limits. In Graph Reasoning,
larger graphs frequently exhausted the token budget before
generating parseable outputs, suppressing positive reward
signals (Figure 5). Counting and Spatial Reasoning, with
shorter outputs, were less affected. For domains requiring
verbose reasoning, token budget allocation may be a more
binding constraint than data size.
Consistent with the multi-factor framing in Section 3.1,
these findings reflect interacting system constraints rather
than effects of a single “complexity” factor. Alternative
contributors, including optimization dynamics, reward spar-
sity, and task-specific reward design, may also shape the
observed patterns.
5
CONCLUSION
In this work, we have presented a systematic characteriza-
tion of open-source SLMs fine-tuned using RLVR under low
data regimes. Using three procedurally generated datasets
with controllable properties (covering number counting
problems, graph reasoning, and spatial reasoning), we char-
acterize the effects of dataset size, diversity, and complexity
on model reasoning capabilities. We find that even within
low data regimes, under RLVR, open-source SLMs trained
on low complexity tasks generalize to higher complexity
tasks not seen during training, and training on a mixture
of complexities is associated with greater accuracy gains
under lower data budgets—specifically, we find that the
mixed complexity setting provides up to 5× the sample
efficiency versus training on easy tasks alone. Our results
also demonstrate that procedural data generation is a useful
tool for starting to understand the data scaling laws that
govern the effectiveness of RLVR, and these findings moti-
vate wider research into the development of empirical laws
that relate reasoning capabilities after fine-tuning to dataset
composition and size.
Although our work begins to characterize the effectiveness
of RLVR in low-data regimes, several limitations should
be noted.
Our findings are specific to a 4B parameter
model with LoRA fine-tuning under fixed low compute
budgets, and we did not perform multi-seed repetitions.
These results are intended to inform practitioners operat-
ing under similar constraints, not to claim universal scaling
laws. While data composition effects may remain relevant
at larger scales (Tan et al., 2025), the quantitative gains (e.g.,
5× sample efficiency) may not directly transfer to larger
models. Additionally, procedural tasks do not capture the
full complexity of real-world data, and we did not evaluate
transfer to natural language benchmarks.
Future work could develop more extensive scaling laws
relating dataset properties to post-fine-tuning performance,
validate these trends at larger model scales and compute
budgets, evaluate transfer to natural language benchmarks,
and develop formal budget-aware RLVR theory that captures
interactions between optimization budget, token limits, and
reward sparsity. As shown by Khatri et al. (2025), this
requires significant computational resources.
REFERENCES
Anthropic.
Claude sonnet 4.5 system card.
Tech-
nical report, Anthropic, September 2025.
Avail-
able
at:
https://assets.anthropic.
com/m/12f214efcc2f457a/original/
Claude-Sonnet-4-5-System-Card.pdf.
Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I.,
Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang,
D., Rosen, E., et al. Gemini 2.5: Pushing the frontier
with advanced reasoning, multimodality, long context,
and next generation agentic capabilities. arXiv preprint
arXiv:2507.06261, 2025.
Dang, Q.-A. and Ngo, C. Reinforcement learning for reason-
ing in small llms: What works and what doesn’t. arXiv
preprint arXiv:2503.16219, 2025.
Denis, M. Space and spatial cognition: A multidisciplinary
perspective. Routledge, 2017.
Dsouza, A., Vishwakarma, H., Qi, Z., Bauer, J., Pham, D.,
Walshe, T., Parchami, A., Sala, F., and Varma, P. Automat-
ing benchmark design. arXiv preprint arXiv:2510.25039,
2025.
Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R.,
Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: In-
centivizing reasoning capability in llms via reinforcement
learning. arXiv preprint arXiv:2501.12948, 2025.
He, Z., Liang, T., Xu, J., Liu, Q., Chen, X., Wang, Y., Song,
L., Yu, D., Liang, Z., Wang, W., et al. Deepmath-103k: A
large-scale, challenging, decontaminated, and verifiable
mathematical dataset for advancing reasoning. arXiv
preprint arXiv:2504.11456, 2025.


## Page 12

Measuring the Effectiveness of RLVR in Low Data and Compute Regimes
Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E.,
Cai, T., Rutherford, E., de Las Casas, D., Hendricks,
L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E.,
Millican, K., van den Driessche, G., Damoc, B., Guy,
A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W.,
Vinyals, O., and Sifre, L.
Training compute-optimal
large language models. arXiv preprint arXiv:2203.15556,
2022.
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang,
S., Wang, L., and Chen, W. LoRA: Low-rank adaptation
of large language models. In International Conference
on Learning Representations (ICLR), 2022.
Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B.,
Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and
Amodei, D. Scaling laws for neural language models.
arXiv preprint arXiv:2001.08361, 2020.
Khatri, D., Madaan, L., Tiwari, R., Bansal, R., Duvvuri,
S. S., Zaheer, M., Dhillon, I. S., Brandfonbrener, D.,
and Agarwal, R. The art of scaling reinforcement learn-
ing compute for llms. arXiv preprint arXiv:2510.13786,
2025.
Lai, H., Liu, X., Gao, J., Cheng, J., Qi, Z., Xu, Y., Yao, S.,
Zhang, D., Du, J., Hou, Z., Lv, X., Huang, M., Dong, Y.,
and Tang, J. A survey of post-training scaling in large
language models. In Proceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics
(ACL), pp. 2486–2511, 2025.
Li, X., Zou, H., and Liu, P. Limr: Less is more for rl scaling.
arXiv preprint arXiv:2502.11886, 2025.
Liu, X., Liang, T., He, Z., Xu, J., Wang, W., He, P., Tu, Z.,
Mi, H., and Yu, D. Trust, but verify: A self-verification ap-
proach to reinforcement learning with verifiable rewards.
arXiv preprint arXiv:2505.13445, 2025.
OpenAI.
Gpt-5 system card.
Technical report,
OpenAI, August 2025a.
Version published Au-
gust 7. Available at: https://cdn.openai.com/
gpt-5-system-card.pdf.
OpenAI.
o3 and o4-mini system card.
Technical
report, OpenAI, April 2025b.
System card avail-
able at: https://deploymentsafety.openai.
com/o3/sabotage.
Poddar, S., Wan, Y., Ivison, H., Gupta, A., and Jaques,
N. Personalizing reinforcement learning from human
feedback with variational preference learning. Advances
in Neural Information Processing Systems, 37:52516–
52544, 2024.
Schulman, J. and Lab, T. M. Lora without regret. Thinking
Machines Lab: Connectionism, 2025. doi: 10.64434/tml.
20250929. https://thinkingmachines.ai/blog/lora/.
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang,
H., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. Deepseek-
math: Pushing the limits of mathematical reasoning in
open language models. arXiv preprint arXiv:2402.03300,
2024.
Shen, W., Liu, G., Wu, Z., Zhu, R., Yang, Q., Xin, C., Yue,
Y., and Yan, L. Exploring data scaling trends and effects
in reinforcement learning from human feedback. arXiv
preprint arXiv:2503.22230, 2025.
Tan, Z., Geng, H., Yu, X., Zhang, M., Wan, G., Zhou, Y., He,
Q., Xue, X., Zhou, H., Fan, Y., Li, Z., Zhang, Z., Zhang,
G., Zhang, C., Yin, Z., Torr, P., and Bai, L. Scaling
behaviors of llm reinforcement learning post-training:
An empirical study in mathematical reasoning. arXiv
preprint arXiv:2509.25300, 2025.
Wang, S., Asilis, J., Akg¨ul, ¨O. F., Bilgin, E. B., Liu, O., and
Neiswanger, W. Tina: Tiny reasoning models via lora.
arXiv preprint arXiv:2504.15777, 2025a.
Wang, Y., Yang, Q., Zeng, Z., Ren, L., Liu, L., Peng, B.,
Cheng, H., He, X., Wang, K., Gao, J., et al. Reinforce-
ment learning for reasoning in large language models with
one training example. arXiv preprint arXiv:2504.20571,
2025b.
Wen, X., Liu, Z., Zheng, S., Ye, S., Wu, Z., Wang, Y., Xu,
Z., Liang, X., Li, J., Miao, Z., Bian, J., and Yang, M.
Reinforcement learning with verifiable rewards implicitly
incentivizes correct reasoning in base llms. arXiv preprint
arXiv:2506.14245, 2025.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D.,
Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang,
J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou,
J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L.,
Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P.,
Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang,
T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan,
Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang,
Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3
technical report. arXiv preprint arXiv:2505.09388, 2025.
Zeng, A., Lv, X., Zheng, Q., Hou, Z., Chen, B., Xie, C.,
Wang, C., Yin, D., Zeng, H., Zhang, J., et al. Glm-4.5:
Agentic, reasoning, and coding (arc) foundation models.
arXiv preprint arXiv:2508.06471, 2025.
Zhang, B., Liu, Z., Cherry, C., and Firat, O. When scaling
meets LLM finetuning: The effect of data, model and
finetuning method. arXiv preprint arXiv:2402.17193,
2024.



---

# Paper 4: PRDP_Proximal_Reward_Difference_Prediction_for_Large-Scale_Reward_Finetuning_of_Diffusion_Models


**Source file:** `PRDP_Proximal_Reward_Difference_Prediction_for_Large-Scale_Reward_Finetuning_of_Diffusion_Models.pdf`

**Pages:** 11


## Page 1

PRDP: Proximal Reward Difference Prediction
for Large-Scale Reward Finetuning of Diffusion Models
Fei Deng1,2*, Qifei Wang1, Wei Wei3†, Tingbo Hou1, Matthias Grundmann1
1Google, 2Rutgers University, 3Accenture
https://fdeng18.github.io/prdp
A painting of a girl standing 
on a mountain looking out 
at an approaching storm 
over the ocean, with wind 
blowing and ocean mist, 
surrounded by lightning.
A night scene of a lavender 
ﬁeld with a town and 
church in the background, 
reminiscent of Vincent van 
Gogh's style.
cinematic still of 
an adorable 
walking robot in 
the desert, at 
sunset
a close up of a cat 
wearing a pikachu 
hat, reddit, gif, real 
life charmander, 
very aesthetic!!!!!!, 
soft!!
rural house with 
a garden and a 
swimming pool
An abandoned 
Segway in the forest
Stable Diffusion v1.4
PRDP
A corgi dressed as a 
bee costume.
Figure 1. Generation samples on complex, unseen prompts. Our proposed method, PRDP, achieves stable black-box reward finetuning
for diffusion models for the first time on large-scale prompt datasets, leading to superior generation quality on complex, unseen prompts.
Here, PRDP is finetuned from Stable Diffusion v1.4 on the training set prompts of Pick-a-Pic v1 dataset, using a weighted combination of
rewards: PickScore = 10, HPSv2 = 2, Aesthetic = 0.05. The images within each column are generated using the same random seed.
Abstract
Reward finetuning has emerged as a promising approach
to aligning foundation models with downstream objectives.
Remarkable success has been achieved in the language do-
main by using reinforcement learning (RL) to maximize re-
wards that reflect human preference. However, in the vi-
sion domain, existing RL-based reward finetuning methods
are limited by their instability in large-scale training, ren-
dering them incapable of generalizing to complex, unseen
prompts. In this paper, we propose Proximal Reward Dif-
ference Prediction (PRDP), enabling stable black-box re-
ward finetuning for diffusion models for the first time on
large-scale prompt datasets with over 100K prompts. Our
key innovation is the Reward Difference Prediction (RDP)
objective that has the same optimal solution as the RL ob-
*Work done during an internship at Google.
†Work done while working at Google.
jective while enjoying better training stability. Specifically,
the RDP objective is a supervised regression objective that
tasks the diffusion model with predicting the reward differ-
ence of generated image pairs from their denoising trajec-
tories. We theoretically prove that the diffusion model that
obtains perfect reward difference prediction is exactly the
maximizer of the RL objective. We further develop an online
algorithm with proximal updates to stably optimize the RDP
objective. In experiments, we demonstrate that PRDP can
match the reward maximization ability of well-established
RL-based methods in small-scale training. Furthermore,
through large-scale training on text prompts from the Hu-
man Preference Dataset v2 and the Pick-a-Pic v1 dataset,
PRDP achieves superior generation quality on a diverse
set of complex, unseen prompts whereas RL-based methods
completely fail.
7423
2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
2575-7075/24/$31.00 ©2024 IEEE
DOI 10.1109/CVPR52733.2024.00709
2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) | 979-8-3503-5300-6/24/$31.00 ©2024 IEEE | DOI: 10.1109/CVPR52733.2024.00709
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:31:20 UTC from IEEE Xplore.  Restrictions apply.


## Page 2

1. Introduction
Diffusion models have achieved remarkable success in gen-
erative modeling of continuous data, especially in photore-
alistic text-to-image synthesis [7, 15, 30, 36, 37, 40, 44, 46].
However, the maximum likelihood training objective of dif-
fusion models is often misaligned with their downstream
use cases, such as generating novel compositions of objects
unseen during training, and producing images that are aes-
thetically preferred by humans.
A similar misalignment problem exists in language mod-
els, where exactly matching the model output to the training
distribution tends to yield undesirable model behavior. For
example, the model may output biased, toxic, or harmful
content. A successful solution, called reinforcement learn-
ing from human feedback (RLHF) [2, 31, 47, 61], is to use
reinforcement learning (RL) to finetune the language model
such that it maximizes some reward function that reflects
human preference. Typically, the reward function is defined
by a reward model pretrained from human preference data.
Inspired by the success of RLHF in language models,
researchers have developed several reward models in the vi-
sion domain [22, 23, 53–55] that are similarly trained to be
aligned with human preference. Furthermore, two recent
works, DDPO [4] and DPOK [10], have explored using RL
to finetune diffusion models. They both view the denoising
process as a Markov decision process [9], and apply policy
gradient methods such as PPO [42] to maximize rewards.
However, policy gradients are notoriously prone to high
variance, causing training instability. To reduce variance, a
common approach is to normalize the rewards by subtract-
ing their expected value [48, 51]. DPOK fits a value func-
tion to estimate the expected reward, showing promising re-
sults when trained on ∼200 prompts. Alternatively, DDPO
maintains a separate buffer for each prompt to track the
mean and variance of rewards, demonstrating stable train-
ing on ∼400 prompts and better performance than DPOK.
Nevertheless, we find that DDPO still suffers from training
instability on larger numbers of prompts, depriving it of the
benefits offered by training on large-scale prompt datasets.
In this paper, we propose Proximal Reward Difference
Prediction (PRDP), a scalable reward maximization algo-
rithm that does not rely on policy gradients. To the best of
our knowledge, PRDP is the first method that achieves sta-
ble large-scale finetuning of diffusion models on more than
100K prompts for black-box reward functions.
Inspired by the recent success of DPO [35] that converts
the RLHF objective for language models into a supervised
classification objective, we derive for diffusion models a
new supervised regression objective, called Reward Differ-
ence Prediction (RDP), that has the same optimal solution
as the RLHF objective while enjoying better training sta-
bility. Specifically, our RDP objective tasks the diffusion
model with predicting the reward difference of generated
image pairs from their denoising trajectories. We prove that
the diffusion model that obtains perfect reward difference
prediction is exactly the maximizer of the RLHF objective.
We further propose proximal updates and online optimiza-
tion to improve training stability and generation quality.
Our contributions are summarized as follows:
• We propose PRDP, a scalable reward finetuning method
for diffusion models, with a new reward difference pre-
diction objective and its stable optimization algorithm.
• PRDP achieves stable black-box reward maximization for
diffusion models for the first time on large-scale prompt
datasets with over 100K prompts.
• PRDP exhibits superior generation quality and general-
ization to unseen prompts through large-scale training.
2. Preliminaries
In this section, we briefly introduce the generative process
of denoising diffusion probabilistic models (DDPMs) [15,
44, 46]. Given a text prompt c, a text-to-image DDPM πθ
with parameters θ defines a text-conditioned image distri-
bution πθ(x0|c) as follows:
πθ(x0|c) =
Z
πθ(x0:T |c) dx1:T
=
Z
p(xT )
T
Y
t=1
πθ(xt−1|xt, c) dx1:T ,
(1)
where x0 is the image, and x1:T are latent variables of the
same dimension as x0. Typically, p(xT ) = N(0, I), and
πθ(xt−1|xt, c) = N(xt−1; µθ(xt, c), σ2
t I)
(2)
is a Gaussian distribution with learnable mean and fixed co-
variance. To generate an image x0 ∼πθ(x0|c), DDPM
uses ancestral sampling. That is, it samples the full de-
noising trajectory x0:T ∼πθ(x0:T |c), by first sampling
xT ∼p(xT ), and then sampling xt−1 ∼πθ(xt−1|xt, c)
for t = T, . . . , 1. Conversely, given a denoising trajectory
x0:T , we can analytically compute its log-likelihood as
log πθ(x0:T |c) = log p(xT ) +
T
X
t=1
log πθ(xt−1|xt, c) (3)
= −1
2
T
X
t=1
∥xt−1 −µθ(xt, c)∥2
σ2
t
+ C, (4)
where C is a constant independent of θ.
3. Method
3.1. Reward Difference Prediction for
KL-Regularized Reward Maximization
We start derivation from the typical RLHF objective [10]:
max
πθ
Ex0,c[r(x0, c) −βKL[πθ(x0|c)||πref(x0|c)]] . (5)
7424
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:31:20 UTC from IEEE Xplore.  Restrictions apply.


## Page 3

xa
T
xa
0
xb
0
xb
T
Denoising Trajectories
⇡✓old
Prompt c
⇡✓
r(xa
0, c)
r(xb
0, c)
ˆr✓(xa
0:T , c)
ˆr✓(xb
0:T , c)
∆ˆr✓
S
XicbZDJSgNBEIZ74h53PXpDIghBlxO4p68BjBLJAEqelUtEn39NBdo4TBl/Cqb+QT+BjexJOd5aAxPxT8fFUFVX+cKukoD+Cwszs3PzC4lJxeWV
1bX1jc6vmTGYFVoVRxjZicKhkglWSpLCRWgQdK6zHvctBv/6I1kmT3FI/xbaG+0R2pQDyqNG6QkXA7d1GKSyHQ/H/JhqbEhurcrcZHLQ6RmQaExIKn
GtGYUrtHCxJofC52MocpiB6cI9NbxPQ6Nr58OBnvudJh3eN9ZUQH9LfGzlo5/o69pMa6MFN9gZwai/W07ExPYLY/Tkr19BDgUpN0EyRtOZp4gXqnr
VzmaQZYSJGH3QzxcnwQay8Iy0KUn1vQFjpQ+DiASwI8uEXfbzRZJj/Te2wHJ2Uj2+OSucX46AX2Q7bZfsYqfsnF2zCqsywR7Ya/sLXgPoOv4Hs0
WgjGO9vsjwozP4x2su0=</latexit>∆r
Predicted Reward Difference
Groundtruth Reward Difference
Model Snapshot
A green
colored rabbit
Diffusion
Model
Diffusion
Model
Reward
Model
MSE
Loss
b
O+au+8vnCmtwJ4wytirDBwqmWOPJCm8KiyCzhReZtOzun95i9ZJk/+gWYEDTe5HEsB5NGw2eproIkAVX2b7/dpgSfhs12HKWd+Esa8zhK0vTwqFOL4xrxJIoX1WbLOh/uBgf9kRGlxpyEAueuk7igQWpFA4b/RLhwWIKdzgtZc5aHSDauF+zj96MuJjY/3LiS/ovxsVaOdmOvOTtVe32qvhi71Mv4yNmRJk7pmtSsMUBSq1QktF0pqfKyfQOB1UMi9Kwlw8XTAuFSfD64z5SFoUpGZegLDSh8DFBCwI8j/R8PH+zZD/X1wcRsnqP9uN09XQa9yfbYB7bPEnbCuwrO2c9JtiM3b
F79iv4HTyGa+HG02gYLHda7FmFzT9sXbZH</latexit>L(✓)
Figure 2. PRDP framework. PRDP mitigates the instability of policy gradient methods by converting the RLHF objective to an equivalent
supervised regression objective. Specifically, given a text prompt, PRDP samples two images, and tasks the diffusion model with predicting
the reward difference of these two images from their denoising trajectories. The diffusion model is updated by stochastic gradient descent
on the MSE loss that measures the prediction error. We prove that the MSE loss and the RLHF objective have the same optimal solution.
Here, we seek to finetune the diffusion model πθ by maxi-
mizing a given reward function r(x0, c) with a KL regular-
ization, whose strength is controlled by a hyperparameter
β. The reward function can be a pretrained reward model
(e.g., HPSv2 [53], PickScore [22]) that measures the gen-
eration quality, and the KL regularization discourages πθ
from deviating too far from the pretrained diffusion model
πref (e.g., Stable Diffusion [37]).
This helps πθ to pre-
serve the overall generation capability of πref, and keeps
the generated images x0 close to the distribution where the
reward model is accurate. The expectation is taken over text
prompts c ∼p(c) and images x0 ∼πθ(x0|c), where p(c)
is a predefined prompt distribution, usually a uniform dis-
tribution over a set of training prompts.
In contrast to language models, the KL regularization
in Eq. (5) cannot be computed analytically, due to the in-
tractable integral defined in Eq. (1). Hence, we instead max-
imize a lower bound of the objective in Eq. (5):
max
πθ
Ex0,c[r(x0, c) −βKL[πθ(¯x|c)||πref(¯x|c)]] ,
(6)
where ¯x := x0:T is the full denoising trajectory. We provide
the proof of lower bound in Appendix A.1.
While it is possible to apply REINFORCE [51] or more
advanced policy gradient methods [4, 10, 42] to optimize
Eq. (6), we empirically find they are hard to scale to large
numbers of prompts due to training instability. Inspired by
DPO [35], we propose to reformulate Eq. (6) into a super-
vised learning objective, allowing stable training on more
than 100K prompts.
First, we derive the optimal solution to Eq. (6) as:
πθ⋆(¯x|c) =
1
Z(c)πref(¯x|c) exp
 1
β r(x0, c)

,
(7)
where Z(c) =
R
πref(¯x|c)exp(r(x0, c)/β)d¯x is the parti-
tion function. Proof can be found in Appendix A.2. Since
Z(c) is intractable, Eq. (7) cannot be directly used to com-
pute πθ⋆. However, it reveals that πθ⋆must satisfy
log πθ⋆(¯x|c)
πref(¯x|c) = 1
β r(x0, c) −log Z(c)
(8)
for all ¯x and c. This allows us to cancel the log Z(c) term
by considering two denoising trajectories ¯xa and ¯xb that
correspond to the same text prompt c:
log πθ⋆(¯xa|c)
πref(¯xa|c) −log πθ⋆(¯xb|c)
πref(¯xb|c) = r(xa
0, c) −r(xb
0, c)
β
.
(9)
Define
ˆrθ(¯x, c) := log πθ(¯x|c)
πref(¯x|c),
(10)
∆ˆrθ(¯xa, ¯xb, c) := ˆrθ(¯xa, c) −ˆrθ(¯xb, c),
(11)
∆r(xa
0, xb
0, c) := r(xa
0, c) −r(xb
0, c),
(12)
then Eq. (9) becomes
∆ˆrθ⋆(¯xa, ¯xb, c) = ∆r(xa
0, xb
0, c)/β.
(13)
This motivates us to optimize πθ by minimizing the follow-
ing mean squared error (MSE) loss:
L(θ) = E¯xa,¯xb,c [lθ(¯xa, ¯xb, c)]
(14)
:= E¯xa,¯xb,c
∆ˆrθ(¯xa, ¯xb, c) −∆r(xa
0, xb
0, c)/β
2 .
We call L(θ) the Reward Difference Prediction (RDP) ob-
jective, since we learn πθ by predicting the reward differ-
ence ∆r(xa
0, xb
0, c) instead of directly maximizing the re-
ward. An illustration is provided in Fig. 2. We further show
in Appendix A.3 that
πθ = πθ⋆⇐⇒L(θ) = 0.
(15)
7425
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:31:20 UTC from IEEE Xplore.  Restrictions apply.


## Page 4

Training w/o Proximal Updates
Training w/ Proximal Updates
Figure 3. Effect of proximal updates. We show generation samples during the PRDP training process. Here, we use the small-scale setup
described in Sec. 4.1 and HPSv2 as the reward model. All samples use the same prompt “A painting of a deer” and the same random seed.
(Left) Without proximal updates, training is quite unstable, and the generation quickly becomes meaningless noise. (Right) With proximal
updates, the training stability is remarkably improved.
Algorithm 1 PRDP Training
Require: pretrained diffusion model πref, training prompt distri-
bution p(c), reward model r(x0, c), training epochs E, gradi-
ent updates K per epoch, prompt batch size N, image batch
size B per prompt
1: πθ ←πref
▷Initialization
2: for epoch e = 1, . . . , E do
3:
πθold ←πθ
▷Model snapshot
4:
{cn}N
n=1
iid
∼p(c)
▷Sample text prompts
5:
for each text prompt cn do
6:
{¯xn,i}B
i=1
iid
∼πθold(¯x|cn)
▷Denoising trajectories
7:
end for
8:
Obtain rewards r(xn,i
0 , cn) for all n, i
9:
for gradient step k = 1, . . . , K do
10:
L(θ) ←
1
N(B
2)
PN
n=1
P
1≤i<j≤B lθ(¯xn,i, ¯xn,j, cn)
11:
Update model parameters θ by gradient descent
12:
end for
13: end for
3.2. Online Optimization
To estimate the expectation in L(θ), we need samples of de-
noising trajectories ¯xa and ¯xb that correspond to the same
prompt c. A straightforward approach, as similarly done in
DPO, is to sample ¯xa, ¯xb iid
∼πref(¯x|c). This can be im-
plemented as uniform sampling from a fixed offline dataset
generated by the pretrained model πref.
However, the offline dataset lacks sufficient coverage of
samples from πθ(¯x|c) that keeps updating, leading to sub-
optimal generation quality. Therefore, we propose an online
optimization procedure, inspired by online RL algorithms.
Specifically, we sample ¯xa, ¯xb iid
∼πθold(¯x|c), where θold is
a snapshot of the diffusion model parameters θ, and we set
θold ←θ every K gradient updates. In practice, we use
πθold to generate a batch of denoising trajectories, and then
use all pairs of denoising trajectories in the batch to com-
pute the loss L(θ). Details are provided in Algorithm 1. We
will show in Sec. 4.3 that online optimization significantly
improves generation quality.
3.3. Proximal Updates for Stable Training
We find in our experiments that directly optimizing Eq. (14)
is prone to training instability, as illustrated in Fig. 3 (Left).
This is likely due to excessively large model updates during
training. To resolve this issue, we propose proximal updates
that remove the incentive for moving πθ too far away from
πθold. Inspired by PPO [42], we achieve this by clipping the
log probability ratio log(πθ(¯x|c)/πθold(¯x|c)) to be within
a small interval [−ϵ′, ϵ′]. This can be implemented by clip-
ping the ˆrθ(¯x, c) as ˆrclip
θ
(¯x, c) :=
clip (ˆrθ(¯x, c), ˆrθold(¯x, c) −ϵ′, ˆrθold(¯x, c) + ϵ′) ,
(16)
because log(πθ(¯x|c)/πθold(¯x|c)) = ˆrθ(¯x, c) −ˆrθold(¯x, c).
We then use ˆrclip
θ
(¯x, c) to compute the clipped MSE loss
lclip
θ
(¯xa, ¯xb, c) :=
∆ˆrclip
θ
(¯xa, ¯xb, c) −∆r(xa
0, xb
0, c)/β

2
,
(17)
where ∆ˆrclip
θ
(¯xa, ¯xb, c) := ˆrclip
θ
(¯xa, c)−ˆrclip
θ
(¯xb, c). Sim-
ilar to PPO [42], our final loss is the maximum of the
clipped and unclipped MSE loss:
lθ(¯xa, ¯xb, c) ←max(lθ(¯xa, ¯xb, c), lclip
θ
(¯xa, ¯xb, c)). (18)
This ensures that we minimize an upper bound of the origi-
nal loss, making the optimization problem well-defined.
In practice, the clipping in Eq. (16) is decomposed and
applied at each denoising step t. First, ˆrθ(¯x, c) can be de-
composed as ˆrθ(¯x, c) = PT
t=1 ˆrθ,t(¯x, c), where
ˆrθ,t(¯x, c) := log(πθ(xt−1|xt, c)/πref(xt−1|xt, c)) . (19)
We apply clipping to each ˆrθ,t(¯x, c) as ˆrclip
θ,t (¯x, c) :=
clip (ˆrθ,t(¯x, c), ˆrθold,t(¯x, c) −ϵ, ˆrθold,t(¯x, c) + ϵ) ,
(20)
where ϵ is the stepwise clipping range. Finally, we replace
Eq. (16) with
ˆrclip
θ
(¯x, c) :=
T
X
t=1
ˆrclip
θ,t (¯x, c).
(21)
As shown in Fig. 3 (Right), our proposed proximal updates
can remarkably improve optimization stability.
7426
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:31:20 UTC from IEEE Xplore.  Restrictions apply.


## Page 5

snake
Reward Model: HPSv2
whale
horse
duck
monkey
goat
Reward Model: PickScore
Stable
Diffusion
DDPO
PRDP
Figure 4. Generation samples from small-scale training. DDPO and PRDP are finetuned from Stable Diffusion v1.4 on 45 prompts
consisting of common animal names, with HPSv2 (Left) and PickScore (Right) as the reward model. Samples within each column use the
same random seed. The prompt template is “A painting of a ⟨animal⟩”, where the ⟨animal⟩is listed on top of each column. All prompts
are seen during training. Both DDPO and PRDP significantly improve the generation quality, with PRDP being slightly better.
4. Experiments
In our experiments, we first verify on a set of 45 prompts
that PRDP can match the reward maximization ability of
DDPO [4], which is based on the well-established PPO [42]
algorithm. We then conduct a large-scale training on more
than 100K prompts from the training set of HPDv2 [53],
showing that PRDP can successfully handle large-scale
training whereas DDPO fails. We further perform a large-
scale multi-reward finetuning on the training set prompts of
Pick-a-Pic v1 dataset [22], highlighting the superior genera-
tion quality of PRDP on complex, unseen prompts. Finally,
we showcase the advantages of our algorithm design, such
as online optimization and KL regularization.
4.1. Experimental Setup
To perform reward finetuning, we need a pretrained diffu-
sion model, a pretrained reward model, and a training set of
prompts. For all experiments, we use Stable Diffusion (SD)
v1.4 [37] as the pretrained diffusion model, and finetune the
full UNet weights. For sampling, during both training and
evaluation, we use the DDPM sampler [15] with 50 denois-
ing steps and a classifier-free guidance [14] scale of 5.0.
Small-scale setup. We use a set of 45 prompts, with the
template “A painting of a ⟨animal⟩”, where the ⟨animal⟩is
taken from the list of common animal names used in DDPO.
Table 1. Reward score comparison on small-scale training.
SD v1.4
DDPO
PRDP
HPSv2
0.2855
0.3398
0.3471
PickScore
0.2179
0.2664
0.2700
We conduct reward finetuning separately for two recently
proposed reward models, HPSv2 [53] and PickScore [22].
We train for 100 epochs, where in each epoch, we sample
32 prompts and 16 images per prompt. The evaluation uses
the same set of prompts as training. We report reward scores
averaged over 256 random samples per prompt.
Large-scale setup. Following DRaFT [6], we use more
than 100K prompts from the training set of HPDv2, and
finetune for HPSv2 and PickScore separately. We train for
1000 epochs. In each epoch, we sample 64 prompts and 8
images per prompt. We evaluate the finetuned model on 500
randomly sampled training prompts, as well as a variety of
unseen prompts, including 500 prompts from the Pick-a-Pic
v1 test set, and 800 prompts from each of the four bench-
mark categories of HPDv2, namely animation, concept art,
painting, and photo. We report reward scores averaged over
64 random samples per prompt.
Large-scale multi-reward setup. We mostly follow the
7427
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:31:20 UTC from IEEE Xplore.  Restrictions apply.


## Page 6

cinematic still of 
highly reﬂective 
stainless steel 
train in the desert, 
at sunset
Reward Model: HPSv2
The image is a 
wooden sculpture of 
a cute robot with cat 
ears, displayed in a 
contemporary art 
gallery.
A chibi frog 
character surﬁng 
at the beach.
An anthropomorphic 
frog wizard wearing a 
cape and holding a 
wand.
Digital art of a cherry 
tree overlooking a 
valley with a waterfall 
at sunset.
A monkey in a blue 
top hat painted in 
oil by Vincent van 
Gogh in the 1800s.
Reward Model: PickScore
Stable
Diffusion
DDPO
PRDP
Figure 5. Generation samples from large-scale training. DDPO and PRDP are finetuned from Stable Diffusion v1.4 on over 100K
prompts from the training set of HPDv2, with HPSv2 (Left) and PickScore (Right) as the reward model. Samples within each column
are generated from the prompt shown on top, using the same random seed. All prompts are unseen during training. PRDP significantly
improves the generation quality over Stable Diffusion, whereas DDPO fails to generate reasonable results.
large-scale setup, except that we use the training set prompts
of Pick-a-Pic v1 dataset, and a weighted combination of re-
wards: PickScore = 10, HPSv2 = 2, Aesthetic = 0.05,
where Aesthetic is the LAION aesthetic score.
Baselines. DDPO [4] and DPOK [10] are the two most
recent RL finetuning methods for black-box rewards. Since
DDPO has demonstrated better performance than DPOK,
we mainly compare to DDPO. To ensure a fair comparison,
we train DDPO and PRDP for the same number of epochs,
with the same number of reward queries per epoch. We also
use the same random seeds to sample images for evaluation.
4.2. Main Results
Small-scale finetuning. We show generation samples from
small-scale finetuning in Fig. 4 and reward scores in Tab. 1.
Both DDPO and PRDP can significantly improve the gen-
eration quality over Stable Diffusion, with more vivid col-
ors and details. Quantitatively, PRDP achieves slightly bet-
ter reward scores than DDPO. This verifies that PRDP can
match the reward maximization ability of well-established
policy gradient methods.
Large-scale finetuning. We present generation samples
from large-scale finetuning in Fig. 5 and reward scores in
Tab. 2. We observe that Stable Diffusion generates images
with relevant content but low quality. Meanwhile, DDPO
fails to give reasonable results. It generates irrelevant, low
quality images or even meaningless noise, leading to lower
reward scores than Stable Diffusion. This is due to the in-
stability of DDPO in large-scale training, which we further
investigate in Appendix B. In contrast, PRDP maintains sta-
bility in the large-scale setup, and significantly improves the
generation quality on both seen and unseen prompts.
Large-scale multi-reward finetuning. We provide gen-
eration samples in Figs. 1 and 11 to 15, and reward scores
in Tab. 3, showing the superior generation quality of PRDP
on a diverse set of complex, unseen prompts.
4.3. Effect of Online Optimization
In this section, we show that online optimization has a great
advantage over offline optimization. To ensure a fair com-
parison, we use the same number of reward queries and gra-
dient updates for both methods. Specifically, following the
small-scale setup, for online training, we use 100 epochs,
where each epoch makes 512 queries to the reward model.
7428
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:31:20 UTC from IEEE Xplore.  Restrictions apply.


## Page 7

Table 2. Reward score comparison on large-scale training.
Reward
Model
Method
Seen Prompts
Unseen Prompts
HPD v2
Training Set
Pick-a-Pic v1
Test Set
HPD v2
Animation
HPD v2
Concept Art
HPD v2
Painting
HPD v2
Photo
HPSv2
SD v1.4
0.2685
0.2665
0.2737
0.2656
0.2654
0.2750
DDPO
0.2464
0.2501
0.2673
0.2558
0.2570
0.2093
PRDP
0.3175
0.3050
0.3223
0.3175
0.3172
0.3159
PickScore
SD v1.4
0.2092
0.2082
0.2111
0.2062
0.2059
0.2172
DDPO
0.2032
0.1992
0.2077
0.2125
0.2124
0.1780
PRDP
0.2424
0.2344
0.2450
0.2441
0.2448
0.2387
Online Optimization for PickScore
Online Optimization for HPSv2
Figure 6. Effect of online optimization. We show generation samples during the PRDP training process, with HPSv2 (Left) and PickScore
(Right) as the reward model. We follow the small-scale training setup. The prompts for the first and the second rows are “A painting of
a squirrel” and “A painting of a bird”, respectively. Samples within each row use the same random seed. It can be observed that online
optimization continually improves the generation quality.
For offline training, we sample 51200 images from the pre-
trained Stable Diffusion, obtain their rewards, and then per-
form the same total number of gradient updates as in online
training. We show generation samples during the online op-
timization process in Fig. 6, and quantitative comparisons in
Fig. 7. We observe that online optimization continually im-
proves the generation quality, achieving significantly better
reward scores than offline optimization.
4.4. Effect of KL Regularization
A common limitation of reward finetuning is reward hack-
ing, where the finetuned diffusion model exploits inaccura-
cies in the reward model, and produces undesired images
with high reward scores. In this section, we show that the
KL regularization in our PRDP formulation can help allevi-
ate this issue. For this purpose, we use the LAION aesthetic
predictor as the reward model. It only takes images as input,
and can be exploited by disregarding text-image alignment.
We follow the small-scale setup, except that we train for
250 epochs and directly use the 45 common animal names
as prompts. As demonstrated in Fig. 8, DDPO, without KL
regularization, is prone to reward hacking. It completely ig-
0
20
40
60
80
100
Online Training Epochs
0.28
0.30
0.32
0.34
0.36
HPSv2
Offline
Online
0
20
40
60
80
100
Online Training Epochs
0.20
0.22
0.24
0.26
0.28
PickScore
Figure 7. Comparison of online and offline optimization. We
evaluate the reward scores of model checkpoints during online op-
timization and the final model obtained by offline optimization.
We follow the small-scale training setup, and optimize the models
for HPSv2 and PickScore separately. Online optimization matches
the performance of offline optimization in ∼10 epochs, and keeps
improving the reward score afterwards.
nores the text prompts and generates similar images for all
prompts. In contrast, PRDP with β = 10 can successfully
preserve the text-image alignment while improving the aes-
thetic quality. More analysis can be found in Appendix C.
7429
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:31:20 UTC from IEEE Xplore.  Restrictions apply.


## Page 8

DDPO
PRDP
cat
butterﬂy
frog
Figure 8. Effect of KL regularization. We show generation sam-
ples from DDPO and PRDP when optimizing the LAION aesthetic
score. We use the small-scale training setup, except that we train
for 250 epochs. Samples within each column are generated from
the prompt shown on top, using the same random seed. DDPO,
without KL regularization, over-optimizes the reward, generating
similar images for all prompts. In contrast, PRDP, formulated with
KL regularization, successfully preserves text-image alignment.
5. Related Work
Diffusion models. As a new class of generative models,
diffusion models [15, 44, 46] have achieved remarkable suc-
cess in a wide variety of data modalities, including images
[7, 17, 30, 36, 37, 39–41], videos [16, 43], audios [25], 3D
shapes [13, 32, 57, 60], and robotic trajectories [1, 5, 18]. To
facilitate control over the content and style of generation, re-
cent works have investigated finetuning diffusion models on
various conditioning signals [11, 19, 20, 27, 28, 38, 45, 58].
However, it remains challenging to adapt diffusion models
to downstream use cases that are misaligned with the train-
ing objective, such as generating novel compositions of ob-
jects unseen during training, and producing images that are
aesthetically preferred by humans. Although classifier guid-
ance [7] can help mitigate this issue, the classifier requires
noisy images as input, making it hard to use off-the-shelf
classifiers such as object detectors and aesthetic predictors
for guidance. In contrast, we finetune the diffusion model to
maximize rewards that reflect downstream objectives. Our
method can work with generic off-the-shelf reward models
that take clean images as input.
Language model learning from human feedback. The
maximum likelihood training objective for language models
tends to yield undesirable model behavior, due to the poten-
tially biased, toxic, or harmful content in the training data.
Reinforcement learning from human feedback (RLHF) has
recently emerged as a successful remedy [2, 3, 12, 26, 29,
31, 47, 52, 61]. Typically, a reward model is first trained
from human preference data (e.g., rankings of outputs from
a pretrained language model). Then, the language model is
finetuned by online RL algorithms (e.g., PPO [42]) to max-
imize the score given by the reward model. More recently,
DPO [35] proposes a supervised learning method that di-
rectly optimizes the language model from preference data,
skipping the reward model training and avoiding the insta-
bility of RL algorithms. Our method is inspired by DPO
and PPO, but designed specifically for diffusion models.
Reward finetuning for diffusion models. Inspired by
the success of RLHF in the language domain, researchers
have developed several reward models in the vision domain
[21–24, 34, 53–56]. Moreover, recent works have explored
using these reward models to improve the generation quality
of diffusion models. A simple approach, called supervised
finetuning [23, 54], is to finetune the diffusion model to-
ward high-reward samples from an offline dataset. Its major
drawback is that the generation quality is limited by the of-
fline dataset. For further improvement, RAFT [8] proposes
an online variant that iteratively re-generates the dataset. A
more direct method for online optimization is to backprop-
agate the reward function gradient through the denoising
process [6, 33, 49, 55]. However, this only works for dif-
ferentiable rewards. For generic rewards, DDPO [4] and
DPOK [10] propose RL finetuning. While they have shown
promising results on small prompt sets, they are unstable
in large-scale training. Our work addresses the training in-
stability issue, achieving stable reward finetuning on large-
scale prompt datasets for generic rewards. Concurrent with
our work, Diffusion-DPO [50] adapts DPO to efficiently
align diffusion models from large-scale offline preference
data, and [59] proposes to stabilize large-scale RL finetun-
ing by combining the diffusion model pretraining loss.
6. Conclusion
This paper presents PRDP, the first black-box reward fine-
tuning method for diffusion models that is stable on large-
scale prompt datasets with over 100K prompts. We achieve
this by converting the RLHF objective to an equivalent su-
pervised regression objective and developing its stable opti-
mization algorithm. Our large-scale experiments highlight
the superior generation quality of PRDP on complex, un-
seen prompts, which is beyond the capability of existing RL
finetuning methods. We also demonstrate that the KL reg-
ularization in the PRDP formulation can help alleviate the
common issue of reward hacking. We hope that our work
can inspire future research on large-scale reward finetuning
for diffusion models.
Acknowledgments
We thank authors of DRaFT [6] for sharing their training
prompts and reward models. We appreciate helpful discus-
sion with Ligong Han, Yanwu Xu, Yaxuan Zhu, Zhonghao
Wang, Yunzhi Zhang, Yang Zhao, and Zhisheng Xiao.
7430
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:31:20 UTC from IEEE Xplore.  Restrictions apply.


## Page 9

References
[1] Anurag Ajay, Yilun Du, Abhi Gupta, Joshua B. Tenenbaum,
Tommi S. Jaakkola, and Pulkit Agrawal. Is conditional gen-
erative modeling all you need for decision making? In In-
ternational Conference on Learning Representations, 2023.
8
[2] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell,
Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort,
Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Ka-
davath, Jackson Kernion, Tom Conerly, Sheer El-Showk,
Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tris-
tan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt,
Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown,
Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and
Jared Kaplan. Training a helpful and harmless assistant with
reinforcement learning from human feedback. arXiv preprint
arXiv:2204.05862, 2022. 2, 8
[3] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda
Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna
Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol
Chen, Catherine Olsson, Christopher Olah, Danny Her-
nandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-
Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey
Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite,
Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas
Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby,
Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec,
Sheer El Showk, Stanislav Fort, Tamera Lanham, Timo-
thy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan
Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann,
Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom
Brown, and Jared Kaplan. Constitutional AI: Harmlessness
from AI feedback. arXiv preprint arXiv:2212.08073, 2022.
8
[4] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and
Sergey Levine.
Training diffusion models with reinforce-
ment learning.
In International Conference on Learning
Representations, 2024. 2, 3, 5, 6, 8, 15, 16, 18
[5] Chang Chen, Fei Deng, Kenji Kawaguchi, Caglar Gulcehre,
and Sungjin Ahn. Simple hierarchical planning with diffu-
sion. In International Conference on Learning Representa-
tions, 2024. 8
[6] Kevin Clark, Paul Vicol, Kevin Swersky, and David J. Fleet.
Directly fine-tuning diffusion models on differentiable re-
wards. In International Conference on Learning Represen-
tations, 2024. 5, 8, 17
[7] Prafulla Dhariwal and Alexander Nichol. Diffusion models
beat GANs on image synthesis. In Advances in Neural In-
formation Processing Systems, 2021. 2, 8
[8] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang,
Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, KaShun
SHUM, and Tong Zhang. RAFT: Reward ranked finetuning
for generative foundation model alignment. Transactions on
Machine Learning Research, 2023. 8
[9] Ying Fan and Kangwook Lee. Optimizing DDPM sampling
with shortcut fine-tuning.
In International Conference on
Machine Learning, 2023. 2
[10] Ying
Fan,
Olivia
Watkins,
Yuqing
Du,
Hao
Liu,
Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Moham-
mad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK:
Reinforcement learning for fine-tuning text-to-image diffu-
sion models. In Advances in Neural Information Processing
Systems, 2023. 2, 3, 6, 8
[11] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik,
Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An
image is worth one word: Personalizing text-to-image gen-
eration using textual inversion. In International Conference
on Learning Representations, 2023. 8
[12] Amelia Glaese,
Nat McAleese,
Maja Tr˛ebacz,
John
Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh,
Laura Weidinger,
Martin Chadwick,
Phoebe Thacker,
Lucy
Campbell-Gillingham,
Jonathan
Uesato,
Po-Sen
Huang,
Ramona Comanescu,
Fan Yang,
Abigail See,
Sumanth Dathathri, Rory Greig, Charlie Chen, Doug Fritz,
Jaume Sanchez Elias, Richard Green, Soˇna Mokrá, Nicholas
Fernando, Boxi Wu, Rachel Foley, Susannah Young, Iason
Gabriel, William Isaac, John Mellor, Demis Hassabis, Ko-
ray Kavukcuoglu, Lisa Anne Hendricks, and Geoffrey Irv-
ing.
Improving alignment of dialogue agents via targeted
human judgements. arXiv preprint arXiv:2209.14375, 2022.
8
[13] Jiatao Gu, Alex Trevithick, Kai-En Lin, Joshua M. Susskind,
Christian Theobalt, Lingjie Liu, and Ravi Ramamoorthi.
NerfDiff: Single-image view synthesis with NeRF-guided
distillation from 3D-aware diffusion. In International Con-
ference on Machine Learning, 2023. 8
[14] Jonathan Ho and Tim Salimans.
Classifier-free diffusion
guidance. In NeurIPS 2021 Workshop on Deep Generative
Models and Downstream Applications, 2021. 5
[15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising dif-
fusion probabilistic models. In Advances in Neural Informa-
tion Processing Systems, 2020. 2, 5, 8
[16] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang,
Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben
Poole, Mohammad Norouzi, David J. Fleet, and Tim Sali-
mans. Imagen video: High definition video generation with
diffusion models. arXiv preprint arXiv:2210.02303, 2022. 8
[17] Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet,
Mohammad Norouzi, and Tim Salimans. Cascaded diffu-
sion models for high fidelity image generation. Journal of
Machine Learning Research, 23(47):1–33, 2022. 8
[18] Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey
Levine. Planning with diffusion for flexible behavior synthe-
sis. In International Conference on Machine Learning, 2022.
8
[19] Jindong Jiang, Fei Deng, Gautam Singh, and Sungjin Ahn.
Object-centric slot diffusion. In Advances in Neural Infor-
mation Processing Systems, 2023. 8
[20] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen
Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic:
Text-based real image editing with diffusion models.
In
CVPR, 2023. 8
[21] Junjie Ke, Keren Ye, Jiahui Yu, Yonghui Wu, Peyman Milan-
far, and Feng Yang. VILA: Learning image aesthetics from
7431
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:31:20 UTC from IEEE Xplore.  Restrictions apply.


## Page 10

user comments with vision-language pretraining. In CVPR,
2023. 8
[22] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Ma-
tiana, Joe Penna, and Omer Levy.
Pick-a-Pic: An open
dataset of user preferences for text-to-image generation. In
Advances in Neural Information Processing Systems, 2023.
2, 3, 5, 16, 17
[23] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins,
Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad
Ghavamzadeh, and Shixiang Shane Gu.
Aligning text-
to-image models using human feedback.
arXiv preprint
arXiv:2302.12192, 2023. 2, 8
[24] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi.
BLIP: Bootstrapping language-image pre-training for unified
vision-language understanding and generation. In Interna-
tional Conference on Machine Learning, 2022. 8
[25] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu,
Danilo Mandic, Wenwu Wang, and Mark D Plumbley. Audi-
oLDM: Text-to-audio generation with latent diffusion mod-
els. In International Conference on Machine Learning, 2023.
8
[26] Hao Liu, Carmelo Sferrazza, and Pieter Abbeel. Chain of
hindsight aligns language models with feedback. In Interna-
tional Conference on Learning Representations, 2024. 8
[27] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and
Daniel Cohen-Or. Null-text inversion for editing real images
using guided diffusion models. In CVPR, 2023. 8
[28] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhon-
gang Qi, Ying Shan, and Xiaohu Qie. T2I-adapter: Learning
adapters to dig out more controllable ability for text-to-image
diffusion models. arXiv preprint arXiv:2302.08453, 2023. 8
[29] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu,
Long Ouyang, Christina Kim, Christopher Hesse, Shan-
tanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang,
Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin But-
ton, Matthew Knight, Benjamin Chess, and John Schulman.
WebGPT: Browser-assisted question-answering with human
feedback. arXiv preprint arXiv:2112.09332, 2021. 8
[30] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh,
Pranav Shyam,
Pamela Mishkin,
Bob Mcgrew,
Ilya
Sutskever, and Mark Chen. GLIDE: Towards photorealis-
tic image generation and editing with text-guided diffusion
models. In International Conference on Machine Learning,
2022. 2, 8
[31] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Car-
roll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini
Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob
Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda
Askell, Peter Welinder, Paul F Christiano, Jan Leike, and
Ryan Lowe. Training language models to follow instructions
with human feedback. In Advances in Neural Information
Processing Systems, 2022. 2, 8
[32] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Milden-
hall. DreamFusion: Text-to-3D using 2D diffusion. In In-
ternational Conference on Learning Representations, 2023.
8
[33] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and
Katerina Fragkiadaki.
Aligning text-to-image diffusion
models with reward backpropagation.
arXiv preprint
arXiv:2310.03739, 2023. 8
[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever.
Learning transferable visual
models from natural language supervision. In International
Conference on Machine Learning, 2021. 8
[35] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn. Direct
preference optimization: Your language model is secretly a
reward model. In Advances in Neural Information Process-
ing Systems, 2023. 2, 3, 8, 13
[36] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu,
and Mark Chen. Hierarchical text-conditional image gener-
ation with CLIP latents. arXiv preprint arXiv:2204.06125,
2022. 2, 8
[37] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Björn Ommer. High-resolution image syn-
thesis with latent diffusion models. In CVPR, 2022. 2, 3, 5,
8, 16, 17
[38] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch,
Michael Rubinstein, and Kfir Aberman. DreamBooth: Fine
tuning text-to-image diffusion models for subject-driven
generation. In CVPR, 2023. 8
[39] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee,
Jonathan Ho, Tim Salimans, David Fleet, and Mohammad
Norouzi. Palette: Image-to-image diffusion models. In ACM
SIGGRAPH 2022 Conference Proceedings, 2022. 8
[40] Chitwan Saharia, William Chan, Saurabh Saxena, Lala
Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour,
Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Sali-
mans, Jonathan Ho, David J Fleet, and Mohammad Norouzi.
Photorealistic text-to-image diffusion models with deep lan-
guage understanding.
In Advances in Neural Information
Processing Systems, 2022. 2
[41] Chitwan Saharia, Jonathan Ho, William Chan, Tim Sali-
mans, David J. Fleet, and Mohammad Norouzi.
Image
super-resolution via iterative refinement. IEEE Transactions
on Pattern Analysis and Machine Intelligence, 45(4):4713–
4726, 2023. 8
[42] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Rad-
ford, and Oleg Klimov. Proximal policy optimization algo-
rithms. arXiv preprint arXiv:1707.06347, 2017. 2, 3, 4, 5, 8,
18
[43] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An,
Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual,
Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman.
Make-A-Video: Text-to-video generation without text-video
data. In International Conference on Learning Representa-
tions, 2023. 8
[44] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan,
and Surya Ganguli.
Deep unsupervised learning using
nonequilibrium thermodynamics. In International Confer-
ence on Machine Learning, 2015. 2, 8
[45] Kihyuk Sohn, Lu Jiang, Jarred Barber, Kimin Lee, Nataniel
Ruiz, Dilip Krishnan, Huiwen Chang, Yuanzhen Li, Irfan
7432
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:31:20 UTC from IEEE Xplore.  Restrictions apply.


## Page 11

Essa, Michael Rubinstein, Yuan Hao, Glenn Entis, Irina
Blok, and Daniel Castro Chin.
StyleDrop: Text-to-image
synthesis of any style. In Advances in Neural Information
Processing Systems, 2023. 8
[46] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Ab-
hishek Kumar, Stefano Ermon, and Ben Poole. Score-based
generative modeling through stochastic differential equa-
tions. In International Conference on Learning Represen-
tations, 2021. 2, 8
[47] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler,
Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and
Paul F Christiano. Learning to summarize with human feed-
back. In Advances in Neural Information Processing Sys-
tems, 2020. 2, 8
[48] Richard S Sutton and Andrew G Barto. Reinforcement learn-
ing: An introduction. MIT press, 2018. 2
[49] Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhil
Naik.
End-to-end diffusion latent optimization improves
classifier guidance. In ICCV, 2023. 8
[50] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou,
Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming
Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model align-
ment using direct preference optimization. In CVPR, 2024.
8
[51] Ronald J Williams. Simple statistical gradient-following al-
gorithms for connectionist reinforcement learning. Machine
learning, 8:229–256, 1992. 2, 3
[52] Jeff Wu, Long Ouyang, Daniel M Ziegler, Nisan Stiennon,
Ryan Lowe, Jan Leike, and Paul Christiano.
Recursively
summarizing books with human feedback. arXiv preprint
arXiv:2109.10862, 2021. 8
[53] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng
Zhu, Rui Zhao, and Hongsheng Li. Human preference score
v2: A solid benchmark for evaluating human preferences of
text-to-image synthesis. arXiv preprint arXiv:2306.09341,
2023. 2, 3, 5, 8, 15, 16, 17
[54] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hong-
sheng Li. Human preference score: Better aligning text-to-
image models with human preference. In ICCV, 2023. 8
[55] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai
Li, Ming Ding, Jie Tang, and Yuxiao Dong.
ImageRe-
ward: Learning and evaluating human preferences for text-
to-image generation.
In Advances in Neural Information
Processing Systems, 2023. 2, 8
[56] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mo-
jtaba Seyedhosseini, and Yonghui Wu. CoCa: Contrastive
captioners are image-text foundation models. Transactions
on Machine Learning Research, 2022. 8
[57] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic,
Or Litany, Sanja Fidler, and Karsten Kreis. LION: Latent
point diffusion models for 3D shape generation. In Advances
in Neural Information Processing Systems, 2022. 8
[58] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding
conditional control to text-to-image diffusion models.
In
ICCV, 2023. 8
[59] Yinan Zhang, Eric Tzeng, Yilun Du, and Dmitry Kislyuk.
Large-scale reinforcement learning for diffusion models.
arXiv preprint arXiv:2401.12244, 2024. 8
[60] Linqi Zhou, Yilun Du, and Jiajun Wu. 3D shape genera-
tion and completion through point-voxel diffusion. In ICCV,
2021. 8
[61] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B
Brown, Alec Radford, Dario Amodei, Paul Christiano, and
Geoffrey Irving. Fine-tuning language models from human
preferences. arXiv preprint arXiv:1909.08593, 2019. 2, 8
7433
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:31:20 UTC from IEEE Xplore.  Restrictions apply.



---

# Paper 5: Using_Explainable_Techniques_to_Enhance_Chain_of_Thoughts_in_LLMs


**Source file:** `Using_Explainable_Techniques_to_Enhance_Chain_of_Thoughts_in_LLMs.pdf`

**Pages:** 2


## Page 1

Using Explainable Techniques to Enhance Chain of
Thoughts in LLMs
1st Nafis Tanveer Islam
Multiscale Networked Systems (MNS)
University of Amsterdam
n.t.islam@uva.nl
2nd Ruoyu Wang
Multiscale Networked Systems (MNS)
University of Amsterdam
ruoyu.wang2@student.uva.nl
3rd Zhiming Zhao *
Multiscale Networked Systems (MNS)
University of Amsterdam
z.zhao@uva.nl
Abstract—With the recent advancements in reasoning capa-
bilities with semantic understanding, Large Language Models
(LLMs) are increasingly mimicking human reasoning processes
and aligning more closely with human values. Yet, unlike how
humans sometimes make decisions intuitively by reasoning or
understanding the underlying logic, LLMs struggle with trans-
parency in their internal reasoning. Similar to how humans often
pause to analyze their reasoning step-by-step to understand their
own decisions better, we propose enabling LLMs to perform
analogous introspective reasoning using a Chain-of-Thought
(CoT) framework powered by SHAP-based explainability. CoT
method guides LLMs to explicitly render intermediate reasoning
steps, effectively simulating a human’s reflective thought process.
Meanwhile, SHAP explainability provides context or logic by
quantifying how each semantic token or feature contributes to the
reasoning, analogous to a human examining which factors most
influenced their choices. Our analysis explores how various train-
ing methods influence their semantic comprehension and, conse-
quently, their ability to communicate their reasoning transpar-
ently. We apply this approach to a specialized environmental and
Earth science ranking dataset—characterized by user feedback
and sparse annotations—and we assess whether LLM-generated
rankings can not only reflect accurate semantic understanding
but also transparently emulate human-like introspection.
Index
Terms—LLM,
Explainability,
Reasoning,
Chain-of-
Thought (CoT).
I. INTRODUCTION
With the ongoing advancements in semantic understanding,
Large Language Models (LLMs) increasingly demonstrate
human-like reasoning patterns, thus aligning with human-like
decision-making processes. In knowledge retrieval systems,
an improved semantic capability for ranking can substantially
enhance the user’s access to relevant information [2]. How-
ever, despite their enhanced accuracy, traditional re-ranking
systems often lack transparency, providing little insight into
the reasoning behind their ranking decisions.
Much like humans who occasionally rely on intuition but
often pause to reflect and analyze their decisions explicitly,
current LLMs tend to produce outcomes without clearly artic-
ulating the underlying logic. The use of traditional recommen-
dation strategies, including Collaborative Filtering (CF) lacks
transparency, providing generic rationales like ”other users
liked this” rather than providing an item-specific reasoning
* Corresponding Author
[3]. This limits user trust and the perceived reliability of
recommendations [4].
To address this challenge, we put forth a comprehensive
reasoning approach where LLMs mimic a human cognitive
process of explicitly reflecting on decisions. Therefore, we
manipulate the Chain-of-Thought (CoT) prompting technique,
enabling the LLM to articulate intermediate reasoning steps
sequentially. This step-by-step introspection makes the se-
mantic logic behind rankings transparent and understandable,
mirroring how humans consciously reflect to justify their
choices. We evaluate our introspective CoT and SHAP-driven
approach using a specialized dataset from ENVRI-Hub-Next 1,
tailored explicitly for knowledge sharing among environmental
and Earth science domains [1].
II. SYSTEM DESIGN AND EXPERIMENTS
A. Methodology
To optimize LLM ranking quality, we explore several
training strategies, namely Supervised Fine Tuning (SFT),
Proximal Policy Optimization (PPO), Direct Policy Optimiza-
tion (DPO), and Reward Modeling—using explicit query-
item ranking pairs. After training, we reduce the model to a
single output neuron, which assigns a relevance score to each
candidate for re-ranking. Finally, we use the trained model
with SHAP explainability to generate attribution scores for
each token. Then we pick the top ten tokens with the highest
attribution scores, and pair them with a prompt so that the
LLM can reason based on its understanding using the tokens.
B. Dataset Collection
The dataset was collected during the ENVRI-Hub-Next
Hackathon, where 12 domain experts queried the ENVRI-Hub
Knowledge Base and re-ranked the top 10 BM25-retrieved
results per query on a scale from 0 (irrelevant) to 9 (highly
relevant). These expert annotations form the supervision data
for training and evaluating our re-ranking models. We used
Direct Policy Optimization(DPO) for training. We split the
top five-ranked responses as positives and the bottom five
as negatives, creating 25 pairwise comparisons per query,
yielding 2,350 training pairs in total.
1https://envri.eu/envri-hub-next/
323
2025 IEEE International Conference on eScience (eScience)
2325-3703/25/$31.00 ©2025 IEEE
DOI 10.1109/eScience65000.2025.00052
2025 IEEE International Conference on eScience (eScience) | 979-8-3315-9145-8/25/$31.00 ©2025 IEEE | DOI: 10.1109/eScience65000.2025.00052
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:28:18 UTC from IEEE Xplore.  Restrictions apply.


## Page 2

TABLE I
COMPARISON OF RANKING PERFORMANCE OF VARIOUS MODELS
Model
NDCG@5
Hits@5
NDCG@10
Hits@10
BM25
0.6434
0.70
0.7193
0.70
LoRA-DPO
Llama-2.7B-hf
0.5470
0.60
0.6592
0.70
Qwen
0.6657
0.60
0.7314
0.70
Reward-model
Llama 3.1
0.5905
0.60
0.6774
0.70
Human
0.8000
0.70
0.8000
0.70
C. Evaluation of Ranking Models
Table I summarizes our offline evaluation of different rank-
ing models on NDCG@5, Hits@5, NDCG@10, and Hits@10
metrics. Table I presents the ranking performance of several
models, including classical baselines, LLM-based approaches,
and human-annotated ground truth. Among the automated
systems, the Qwen achieves the highest ranking effectiveness,
outperforming all others in both NDCG@5 and NDCG@10.
Notably, Qwen attains an NDCG@10 of 0.7314, closely ap-
proaching human performance (0.8) and demonstrating strong
alignment with relevance-based ranking expectations.
In contrast, the LoRA-DPO variant of LLaMA-2.7B un-
derperforms across all metrics, suggesting that lightweight
parameter-efficient fine-tuning may not be sufficient for cap-
turing nuanced semantic relationships in re-ranking tasks. The
Reward-model LLaMA 3.1 performs moderately, improving
over LoRA-DPO but still falling short of Qwen. Interestingly,
BM25 remains competitive in terms of Hits5 and Hits10 (both
at 0.70), showing its strength in retrieving relevant items.
Initial Query: “Temperature in the Alps in 2022”
Top Search Result Types: Climatic reports, station-
based time series from Alpine meteorological stations
SHAP Attributions (Summed attribution values of
the tokens normalized to sum to 1):
• Historical station measurements: 0.45
• Temporal proximity to 2022: 0.30
• Geographical relevance (Alps region): 0.20
Chain-of-Thought Explanation: Our system primar-
ily verifies whether a result includes station-level tem-
perature data for the year 2022 (highest contribution:
0.45). Then it checks whether the temporal coverage
aligns with the query year (0.30). Finally, a check for
Alpine region specificity (0.20). A small contribution
is attributed to metadata quality (0.05).
D. Query-Level Qualitative Analysis with CoT and SHAP
Explanations
We further illustrate our approach using two representative
sample queries, retrieved from the ENVRI search interface (top
5 results). For each case, we show Chain-of-Thought (CoT)
reasoning generated by the LLM, augmented with SHAP
values produced by our reward model to explain the ranking
decisions.
Example Query Reasoning: “tsunamis happened in
Sicily since 2000”
Top Search Result Types: Reports on the 2002
Stromboli tsunami, tsunami near Palermo
Simulated SHAP Attributions (Summed attribution
values of the tokens normalized to sum to 1):
• Mention of known event (e.g., 2002 Stromboli
tsunami): 0.50
• Date ≥2000: 0.30
• Precise geographic mention (Sicily coast): 0.15
Chain-of-Thought Explanation: The system initially
detects references to tsunami events in Sicily that
occurred after 2000 (most significant factor: 0.50).
Followed by it confirms temporal alignment with the
query (0.30), and checks for geographic specificity to
the Sicilian coast (0.15). In particular, a minor contri-
bution is attributed to source credibility and scientific
rigor (0.05).
III. CONCLUSION
Our evaluation depicts in detail how the proposed method
combines SHAP attributions and explicit chain-of-thought rea-
soning to provide interpretable explanations. This helps users
gain transparency into why a specific result is ranked higher,
not only via a numerical score, but through a clear and logical
textual reasoning path grounded by relevant features. This
approach enhances trust and facilitates users’ understanding
of the ranking beyond a black-box ranking score.
ACKNOWLEDGMENT
The work was made possible through funding from several
European Union projects: ENVRI-Hub Next (101131141), EV-
ERSE (101129744), BlueCloud-2026 (101094227), OSCARS
(101129751). This research was partially funded by the Dutch
Research Council (NWO) Large-Scale Research Infrastruc-
tures (LSRI) programme for the LTER-LIFE (http://www.lter-
life.nl) infrastructure (grant 184.036.014).
REFERENCES
[1] Siamak Farshidi, Xiaofeng Liao, Na Li, Doron Goldfarb, Barbara Ma-
gagna, Markus Stocker, Keith Jeffery, Peter Thijsse, Christian Pichot,
Andreas Petzold, et al. Knowledge sharing and discovery across hetero-
geneous research infrastructures. Open Research Europe, 1:68, 2023.
[2] Jingtong Gao, Xiangyu Zhao, Muyang Li, Minghao Zhao, Runze Wu,
Ruocheng Guo, Yiding Liu, and Dawei Yin. Smlp4rec: an efficient all-
mlp architecture for sequential recommendations. ACM Transactions on
Information Systems, 42(3):1–23, 2024.
[3] Jonathan L Herlocker, Joseph A Konstan, and John Riedl. Explaining
collaborative filtering recommendations. In Proceedings of the 2000 ACM
conference on Computer supported cooperative work, pages 241–250,
2000.
[4] Yuhan Li, Xinni Zhang, Linhao Luo, Heng Chang, Yuxiang Ren, Irwin
King, and Jia Li.
G-refer: Graph retrieval-augmented large language
model for explainable recommendation. In Proceedings of the ACM on
Web Conference 2025, pages 240–251, 2025.
324
Authorized licensed use limited to: Sathyabama Institute of Science and Technology. Downloaded on August 01,2026 at 03:28:18 UTC from IEEE Xplore.  Restrictions apply.

