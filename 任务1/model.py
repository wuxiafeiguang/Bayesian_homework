"""任务 1：用独立的 Beta 先验和二项似然刻画版本 A、B 的点击率。"""

from dataclasses import dataclass

from scipy.stats import beta, binom


@dataclass(frozen=True)
class ClickRateModel:
    visitors: int
    clicks: int
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def __post_init__(self) -> None:
        if type(self.visitors) is not int or self.visitors < 1:
            raise ValueError("访问人数必须是正整数")
        if type(self.clicks) is not int or not 0 <= self.clicks <= self.visitors:
            raise ValueError("点击人数必须是介于 0 和访问人数之间的整数")
        if not 0 < self.prior_alpha < float("inf") or not 0 < self.prior_beta < float("inf"):
            raise ValueError("Beta 先验参数必须是有限正数")

    @property
    def observed_rate(self) -> float:
        return self.clicks / self.visitors

    def prior_density(self, click_rate: float) -> float:
        return float(beta.pdf(click_rate, self.prior_alpha, self.prior_beta))

    def likelihood(self, click_rate: float) -> float:
        if not 0 <= click_rate <= 1:
            raise ValueError("点击率必须介于 0 和 1 之间")
        return float(binom.pmf(self.clicks, self.visitors, click_rate))


MODEL_A = ClickRateModel(visitors=1200, clicks=96)
MODEL_B = ClickRateModel(visitors=1200, clicks=138)


def main() -> None:
    for version, model in (("A", MODEL_A), ("B", MODEL_B)):
        print(f"版本 {version}")
        print(f"  访问 {model.visitors} 人，点击 {model.clicks} 人，"
              f"观测点击率 {model.observed_rate:.2%}")
        print(f"  先验 Beta({model.prior_alpha:g}, {model.prior_beta:g})")
        print(f"  似然 P(X = {model.clicks} | p)，其中 X | p ~ Binomial({model.visitors}, p)")


if __name__ == "__main__":
    main()
