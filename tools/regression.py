# !/usr/bin/env python3
# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import numpy as np, torch
from typing import TypeVar, Literal

Tensor = torch.Tensor
Parameter = torch.nn.Parameter
T = TypeVar("T")


def isnan(x: T) -> bool:
    """
    Check if the input is NaN (Not a Number).
    """
    if isinstance(x, Tensor):
        return torch.isnan(x).any().item()
    elif isinstance(x, float):
        return np.isnan(x)
    elif isinstance(x, np.ndarray):
        return np.isnan(x).any()
    else:
        raise TypeError(f"Unsupported type for isnan check: {type(x)}")


def exp(x: T) -> T:
    if isinstance(x, Tensor):
        return torch.exp(x)
    else:
        return np.exp(x)


def to_float(x: T) -> float:
    if isinstance(x, Tensor):
        return x.item()
    elif isinstance(x, float):
        return x
    else:
        raise TypeError(f"Unsupported type for conversion to float: {type(x)}")


class Model(torch.nn.Module):
    k1 = Parameter(torch.tensor(1.0, dtype=torch.float32))
    k2 = Parameter(torch.tensor(1.0, dtype=torch.float32))
    k3 = Parameter(torch.tensor(1.0, dtype=torch.float32))
    t1 = Parameter(torch.tensor(0.0, dtype=torch.float32))
    t2 = Parameter(torch.tensor(0.0, dtype=torch.float32))
    t3 = Parameter(torch.tensor(0.0, dtype=torch.float32))
    p1 = Parameter(torch.tensor(1.0, dtype=torch.float32))
    p2 = Parameter(torch.tensor(1.0, dtype=torch.float32))
    p3 = Parameter(torch.tensor(1.0, dtype=torch.float32))

    def check(self):
        if any((isnan(v) for v in self.parameters())):
            raise ValueError("Model parameters contain NaN values.")

    def __dict__(self) -> dict[str, Parameter]:
        return {
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "t1": self.t1,
            "t2": self.t2,
            "t3": self.t3,
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
        }

    @staticmethod
    def freeze(*args: Parameter):
        for p in args:
            p.requires_grad = False

    @staticmethod
    def relax(*args: Parameter):
        for p in args:
            p.requires_grad = True

    def parameters(
        self, inputs=[], overrides: dict[str, float] = dict(), pick: list[str] = None
    ):
        d = self.__dict__()
        if pick is None:
            pick = d.keys()
        elif any(k not in d for k in pick):
            raise ValueError(f"Invalid parameter key: {pick}")
        parameters = [(d | overrides)[k] for k in pick]
        if all((isinstance(i, float) for i in inputs)):
            return (to_float(t) for t in parameters)
        else:
            return parameters

    def __call__(self, L: T, R: T, E: T, **overrides: float) -> T:
        """
        Model function for regression.
        L * R = E
        (k1 * (L^a) + t1) * (k2 * (R^b) + t2) = (k3 * (E^g) + t3)
        """
        g1 = self.group(1, L, **overrides)
        g2 = self.group(2, R, **overrides)
        g3 = self.group(3, E, **overrides)
        return g1 * g2 - g3

    def group(self, n: Literal[1, 2, 3], X: T, **overrides: float) -> T:
        k, t, p = self.parameters([X], overrides, [f"k{n}", f"t{n}", f"p{n}"])
        return k * (X**p) + t

    @torch.no_grad()
    def L(self, R: T, E: T, **overrides: float) -> T:
        g2 = self.group(2, R, **overrides)
        g3 = self.group(3, E, **overrides)
        k, t, p = self.parameters([R, E], overrides, ["k1", "t1", "p1"])
        G = g3 / g2
        return ((G - t) / k) ** (1 / p)

    @torch.no_grad()
    def R(self, L: T, E: T, **overrides: float) -> T:
        g1 = self.group(1, L, **overrides)
        g3 = self.group(3, E, **overrides)
        k, t, p = self.parameters([L, E], overrides, ["k2", "t2", "p2"])
        G = g3 / g1
        return ((G - t) / k) ** (1 / p)

    @torch.no_grad()
    def E(self, L: T, R: T, **overrides: float) -> T:
        g1 = self.group(1, L, **overrides)
        g2 = self.group(2, R, **overrides)
        k, t, p = self.parameters([L, R], overrides, ["k3", "t3", "p3"])
        G = g1 * g2
        return ((G - t) / k) ** (1 / p)


class Regression(Model):
    def L1_loss(self, pred: Tensor, gt: Tensor | None = None):
        """
        Computes the mean absolute error loss (MAE, a.k.a L1 loss)
        """
        if gt is None:
            diff = pred
        else:
            diff = pred - gt
        return torch.mean(torch.abs(diff))

    def L2_loss(self, pred: Tensor, gt: Tensor | None = None):
        """
        Computes the mean square error loss (MSE, a.k.a L2 loss)
        """
        if gt is None:
            diff = pred
        else:
            diff = pred - gt
        return torch.mean(diff**2)

    def forward(self, *args: T, **overrides: T) -> T:
        """
        Model function for regression.
        """
        return self(*args, **overrides)

    def train(self, *args, learning_rate=0.0001, **overrides: float):
        """
        Trains the regression model using gradient descent.
        """
        optimizer = torch.optim.Adam(self.__dict__().values(), lr=learning_rate)

        while True:
            optimizer.zero_grad()
            pred = self.forward(*args, **overrides)
            loss = self.L1_loss(pred)
            yield loss.item()
            loss.backward()
            optimizer.step()
            try:
                self.check()
            except ValueError as e:
                print(f"Training error: {e}")
                plt.pause(1000)


if __name__ == "__main__":

    def get_data():
        from sys import stdin, stderr

        for line in stdin:
            try:
                x, y = map(float, line.strip().split())
                yield (x, y)
            except ValueError:
                print(f"Invalid input line: {line.strip()}", file=stderr)

    x, y, epsilon = torch.tensor(
        [(x, y, 0) for x, y in get_data()] + [(1, 1, 1)],
        requires_grad=False,
        dtype=torch.float32,
    ).T
    model = Regression()

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from colorsys import hsv_to_rgb

    fig, ax = plt.subplots(figsize=(8, 6))

    L = torch.linspace(0, 1, 100)

    prd = list[tuple[Line2D, float]]()

    with torch.no_grad():
        for eps in np.linspace(0, 1.0, 11, endpoint=True):
            R = model.R(L, eps)
            color = hsv_to_rgb(eps / 2, 0.6, 0.8)
            (l,) = ax.plot(L.numpy(), R.numpy(), label=f"ε={eps:.2f}", color=color)
            prd.append((l, eps))


    ax.plot(x.numpy()[:-1], y.numpy()[:-1], "k--", label="Random Walk")

    @torch.no_grad()
    def update_prd():
        for l, eps in prd:
            R = model.R(L, eps)
            l.set_ydata(R.detach().numpy())
        plt.draw()
        plt.pause(0.001)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid()
    plt.show(block=False)
    plt.pause(0.01)

    counter = 0
    min_loss = float("inf")
    fluctuation_counter = 0
    try:
        for loss in model.train(x, y, epsilon):
            counter += 1
            if loss < min_loss:
                fluctuation_counter = 0
            else:
                fluctuation_counter += 1
                if fluctuation_counter > 100:
                    print(
                        f"Loss has not improved for {fluctuation_counter} iterations."
                    )
                    break
            if counter % 100 == 0:
                print(f"Current loss: {loss:.6f}")
                update_prd()
    except KeyboardInterrupt:
        pass

    print(f"\nTraining interrupted after {counter} iterations.")
    print(f"Loss: {loss}")
    for k, v in model.__dict__().items():
        print(f"{k.ljust(2)} = {v.item()}")
