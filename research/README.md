# Research overview

Adaptive Hierarchical Event Memory (AHEM) investigates how a model can use earlier, causally observed network-flow transitions when predicting the next packet's size and per-flow arrival delay.

In a frozen experiment on a selected subset of CSE-CIC-IDS2018, AHEM v6.1 had lower weighted forecasting loss than a similarly sized flow-lane LSTM. The compared models did not receive identical recent global inputs. The selected sample covered returning flows with eligible memory, not every packet or network.

A later, larger experimental model has been evaluated in exploratory local comparisons. Its source provenance, fallback behavior and inference cost limit the conclusions that can currently be drawn. Neither result establishes detection accuracy or production readiness.

The technical manuscript is undergoing review and is not included in this overview. A full paper should describe methods, benchmarks and limitations together when it is ready for public distribution.
