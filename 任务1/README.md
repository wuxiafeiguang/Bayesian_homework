# 任务 1 点击率的贝叶斯建模

## 数据与参数

令 $p_A$ 和 $p_B$ 分别表示版本 A、B 的真实点击率。每名访问用户是否点击「进入课程」用一个二元结果表示。作业给出的汇总数据如下。

| 版本 | 访问人数 $n$ | 点击人数 $x$ | 未点击人数 $n-x$ | 观测点击率 $x/n$ |
| --- | ---: | ---: | ---: | ---: |
| A | 1200 | 96 | 1104 | 8.00% |
| B | 1200 | 138 | 1062 | 11.50% |

## 先验分布

题目未提供历史点击率或其他先验信息，因此两组都采用同一个均匀先验，并假设两个参数先验独立。

$$
p_A \sim \operatorname{Beta}(1,1), \qquad p_B \sim \operatorname{Beta}(1,1).
$$

$\operatorname{Beta}(1,1)$ 的密度在 $[0,1]$ 上为 1，因此先验对两个版本的点击率采用相同的设定。

## 似然函数

假设每组内的用户共享同一个点击概率。给定点击率后，用户点击相互独立，两组的点击人数也条件独立。用 $X_A$、 $X_B$ 表示两组的点击人数。

$$
X_A \mid p_A \sim \operatorname{Binomial}(1200,p_A), \qquad
X_B \mid p_B \sim \operatorname{Binomial}(1200,p_B).
$$

将实际点击人数代入二项分布，得到关于两组点击率的似然函数。

$$
\begin{aligned}
L_A(p_A) &= \binom{1200}{96}p_A^{96}(1-p_A)^{1104}, \\
L_B(p_B) &= \binom{1200}{138}p_B^{138}(1-p_B)^{1062}.
\end{aligned}
$$

联合似然为 $L_A(p_A)L_B(p_B)$。Beta 先验与二项似然共轭，可在任务 2 中分别更新两个版本的后验分布。

## Python 实现

[`model.py`](model.py) 中的 `MODEL_A` 和 `MODEL_B` 保存各组的观测数据和先验参数。`prior_density(click_rate)` 计算给定点击率处的先验密度，`likelihood(click_rate)` 计算该点击率下恰好观测到实际点击人数的概率。`observed_rate` 返回样本点击比例。
