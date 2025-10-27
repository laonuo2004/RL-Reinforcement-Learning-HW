#set page(margin: (top: 2.54cm, bottom: 2.54cm, left: 3.17cm, right: 3.17cm))
#set text(font: ("Times New Roman", "Source Han Serif SC"), size: 12pt)
#set par(first-line-indent: (amount: 2em, all: true))

// 缩进函数：输入缩进距离（em），返回带缩进的块
#let indent-block(amount, content) = {
  block(inset: (left: amount))[
    #content
  ]
}

// 设置标题样式
#set heading(numbering: (..nums) => {
  let level = nums.pos().len()
  if level == 1 {
    // 一级标题：1, 2, 3...
    numbering("1 ", ..nums)
  } else if level == 2 {
    // 二级标题：1.1, 1.2, 1.3...
    let parent = nums.pos().first()
    let current = nums.pos().last()
    numbering("1.", parent)
    numbering("1 ", current)
  }
})

// 设置标题字体大小和粗体
#show heading.where(level: 1): it => {
  set text(size: 18pt, weight: "bold")
  it
  v(1em)
}

#show heading.where(level: 2): it => {
  set text(size: 16pt, weight: "bold")
  it
  v(1em)
}

#set enum(numbering: "(i)")

#align(center)[
  #text(size: 20pt)[2025-2026-1 学期强化学习课程 - 第二次作业]
  
  #text(size: 14pt)[1120231863 #h(1em) 左逸龙]
  
  #text(size: 14pt)[#datetime.today().display("[month repr:long], [day] [year]")]
]

#v(3em)

= 通勤北理工

1. 既然我们已经知晓了最优的 $Q^*$ 表，那么每一状态下的最优策略满足：

$
  pi^*(a|s) = cases(
    1 "if" a = "argmax"_(a in A(s)) Q^*(s, a),
    0 "otherwise",
  )
$

因此最优策略为：

#indent-block(2em)[
- 在状态 $S_1$ 下，乘坐班车
- 在状态 $S_2$ 下，乘坐班车
- 在状态 $S_3$ 下，乘坐地铁
]

#v(1em)

2. 此时最优策略为：

#indent-block(2em)[
- 在状态 $S_(12)$ 下，乘坐班车
- 在状态 $S_3$ 下，乘坐班车
]

显然，此时得到的最优策略与使用真实三状态表示时得到的最优策略不同，关键区别在于原来状态 $S_3$ 的最优策略为*乘坐地铁*，而现在状态 $S_3$ 的最优策略为*乘坐班车*。

之所以会出现这样的变化，是因为*状态聚合*导致智能体 Agent 无法区分$S_1$与$S_2$的*未来价值*，使得决策依据退化为*即时奖励*。分析如下：

#indent-block(2em)[
- Q-learning 算法的更新公式为：
]

$
Q(s, a) = Q(s, a) + alpha [R + gamma max_(a in A(s')) Q(s', a)]
$

其中$alpha$为学习率，$R$为即时奖励，$gamma$为折扣因子。由公式可以看出，决定 Q 值的不仅仅只有*即时奖励*，还有*未来价值的预期*。

#indent-block(2em)[
- 在原始的三状态模型中，Agent 可以精确地知道每个动作会导向哪个具体的状态。从 $S_3$ 出发，乘坐地铁会到达 $S_2$，而 $S_2$ 的长期价值 $(V^*(S_2) = 1.95)$ 远高于乘坐班车所到达的 $S_1$ 的长期价值 $(V^*(S_1) = 1.65)$。尽管坐地铁的即时奖励更低，但为了追求 $S_2$ 带来的更高未来收益，最优策略是选择乘坐地铁。

- 在聚合后的二状态模型中，$S_1$ 和 $S_2$ 被合并为宏状态 $S_(12)$。此时，无论从 $S_3$ 出发选择乘坐班车（到达 $S_1$）还是乘坐地铁（到达 $S_2$），在 Agent 看来，下一个状态都是*同一个* $S_(12)$。因此，这两个动作所带来的未来价值预期是完全相同的（都等于 $gamma V^*(S_(12))$）。

- 当两个动作的*未来价值预期相同时*，决策的优劣就完全取决于*即时奖励*。根据表1，$R(S_3, "班车") = -0.5$，而 $R(S_3, "地铁") = -0.7$。由于 $-0.5 > -0.7$，选择乘坐班车能获得更好的即时奖励。因此，在这种信息受限的情况下，最优策略从乘坐地铁转变为乘坐班车。
]

综上，导致这种策略上变化的原因是状态表示的粒度变粗后，Q-learning 算法泛化（或平均化）了不同状态的价值，导致决策依据从*长远未来价值*退化为*即时奖励*。

#v(1em)

= Frozenlake 小游戏

1. 
