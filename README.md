# Comp-Geom Simplex Implementation

Simplex Method implementation for Caleb and I's MATH 437/COMP 494 project. For a breakdown of how the code works and the algorithms used, check the paper.

To use our code, you'll need numpy installed. At the top of your document, write 

```python
import numpy as np
from simplex_implementation import simplex
```

The `simplex` method takes 8 possible inputs, only one of which is required. Those inputs are:

| Input | Description | Optional or Required? |
|-------|-------------|-----------------------|
| `c` | The coefficeints for the objective function, $c$ | Required |
| `A_lt` | The matrix $A_\leq$ for any $\leq$ constraints | Optional, `None` by default |
| `b_lt` | The vector $b_\leq$ for any $\leq$ constraints | Optional, `None` by default |
| `A_eq` | The matrix $A_=$ for any $=$ constraints | Optional, `None` by default |
| `b_eq` | The vector $b_=$ for any $=$ constraints | Optional, `None` by default |
| `A_gt` | The matrix $A_\geq$ for any $\geq$ constraints | Optional, `None` by default |
| `b_gt` | The vector $b_\geq$ for any $\geq$ constraints | Optional, `None` by default |
| `min` | if you want to solve a minimzation problem | Optional, `False` by default |

You are required to input at least one $A$ and the corresponding $b$, though you are free to input more.

When run, the code will attempt to solve for $x$ given the LP
$$
\begin{alignat*}{2}
    \text{maximize} \quad && c^t \vec{x} \\
    \text{subject to} \quad && A_\leq \vec{x} &\leq b_\leq \\
    && A_= \vec{x} &= b_= \\
    && A_\geq \vec{x} &\geq b_\geq
\end{alignat*}
$$

If `min` is set to true, it will instead try to minimize the objective, not maximize it.

`simplex` will return
* An numpy array of numbers corresponding to the solutions to the LP if possible.
* A numpy array containing `inf` if the problem is unbounded.
* A numpy array full of `NaN` if the problem is infeasible.

It will throw an error if
* The input is invalid (Dimensions dont match).
* There is no problem (`A_lt`, `A_eq`, and `A_gt` are all not defined).

While running the function, you may see
```
RuntimeWarning: divide by zero encountered in divide ratios = tab[1:, -1]/tab[1:, i] # Get ratios
```
This is normal, and the code will still run fine.

Example calls can be found in [simplex_examples.ipynb](simplex_examples.ipynb).

If you have any questions about the code or how to use it, email Caleb or I.
