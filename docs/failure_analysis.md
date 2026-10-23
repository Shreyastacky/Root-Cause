# Two measured baseline failure cases

### re1ob_checkoutservice_loss_1

The annotated root `checkoutservice` ranks **5**. The services/candidates above it are `paymentservice`, `recommendationservice`, `cartservice`, `productcatalogservice`.

| service               | winning_metric   |   score |   pre_std_of_winner |
|:----------------------|:-----------------|--------:|--------------------:|
| paymentservice        | latency-90       | 377.07  |          2.8107e-05 |
| recommendationservice | mem              | 191.56  |       1183.6        |
| cartservice           | latency-90       | 144.7   |          9.34e-05   |
| productcatalogservice | latency-90       |  59.02  |          6.0153e-06 |
| checkoutservice       | mem              |  20.336 |          1.4034e+05 |

**Observed:** the maximum standardized mean shift is larger for these candidates. **Possible explanation:** packet loss can affect callers and request/error metrics, while unrelated resource drift can also score highly. The heuristic does not model dependencies or temporal propagation. These measurements alone do not establish which non-root changes were caused by the injected fault.


### re1ob_checkoutservice_loss_5

The annotated root `checkoutservice` ranks **5**. The services/candidates above it are `productcatalogservice`, `paymentservice`, `emailservice`, `recommendationservice`.

| service               | winning_metric   |   score |   pre_std_of_winner |
|:----------------------|:-----------------|--------:|--------------------:|
| productcatalogservice | latency-90       | 20.499  |          1.8333e-05 |
| paymentservice        | latency-90       | 10.623  |          2.9521e-05 |
| emailservice          | latency-90       |  7.7992 |          2.7547e-05 |
| recommendationservice | mem              |  7.5588 |      17362          |
| checkoutservice       | latency-50       |  4.1954 |          0.078675   |

**Observed:** the maximum standardized mean shift is larger for these candidates. **Possible explanation:** packet loss can affect callers and request/error metrics, while unrelated resource drift can also score highly. The heuristic does not model dependencies or temporal propagation. These measurements alone do not establish which non-root changes were caused by the injected fault.
