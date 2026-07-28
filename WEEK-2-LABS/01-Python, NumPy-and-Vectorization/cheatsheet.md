# NumPy Cheat Sheet — Week 2 Lab (Vectors + Matrices)

## Shapes — the core mental model

| Shape | Meaning | Example |
|---|---|---|
| `()` | scalar — zero dimensions, bare number | `24` |
| `(n,)` | 1D vector — n elements | `[1 2 3 4]` |
| `(m, n)` | 2D matrix — m rows, n columns | `[[1,2],[3,4]]` |

- The comma in `(4,)` matters — it means "tuple with one item," not just the number 4.
- `(1,)` (array with one element, e.g. `[24]`) is **not the same** as `()` (a bare scalar, `24`). You can index into `(1,)`, you can't index into `()`.

## dtype rules

- A NumPy array holds **ONE dtype for all elements** — never mixed.
- If values conflict (some int, some float), NumPy **upcasts to float** (safer direction — float never loses info going from int, but int truncates float).
- `np.zeros()`, `np.random.rand()` → always default to `float64`, regardless of input.
- `np.array([...])` → dtype is **inferred** from the values you give it.

## Indexing

- Indexing starts at **0**, not 1.
- Negative indices count from the end: `a[-1]` = last element.
- Out-of-range index → `IndexError` (Python bounds-checks; it never segfaults).
- **2D array, ONE index** (`a[2]`) → returns the **whole row**, collapses to 1D.
- **2D array, TWO indices** (`a[2, 0]`) → returns **one element**, a scalar.

## Slicing — `a[start:stop:step]`

- `stop` is **always exclusive** (never included).
- Omit `start` → defaults to 0 (beginning).
- Omit `stop` → defaults to end.
- Omit `step` → defaults to 1.
- `a[:]` → everything, unchanged.
- 2D: `a[:, 2:7]` → **all rows**, columns 2–6 → stays 2D.
- 2D: `a[0, 2:7]` → **one row only**, columns 2–6 → collapses to 1D.
- ⚠️ Slices are **views**, not copies — modifying a slice can modify the original array. Use `.copy()` for an independent copy.

## Operations — element-wise vs collapsing

| Operation | Behavior | Result |
|---|---|---|
| `-a`, `a**2`, `a+b`, `5*a` | element-wise (touches every element) | same shape as input |
| `np.sum(a)`, `np.mean(a)` | collapses whole array | scalar `()` |
| `np.dot(a, b)` | multiply element-wise, THEN sum | scalar `()` |

- Vector-vector element-wise ops (`a + b`) require **matching shapes** (same length) — mismatched shapes → `ValueError`.
- Matching **dtype** is NOT required for element-wise ops — int + float of same shape works fine, result upcasts to float.
- Dot product is **commutative**: `np.dot(a,b) == np.dot(b,a)`.

## Why vectorization is faster

- Python for-loops pay interpreter overhead on **every iteration**.
- `np.dot()` / vectorized ops hand the work to optimized C code using **SIMD** hardware — many multiply-adds done in parallel per CPU cycle.
- Real measured example from this lab: ~2000ms (loop) vs ~5ms (vectorized) on 10M elements — roughly 400x faster.

## Course 1 connection — the pattern that matters most

- `X_train` → shape `(m, n)`: `m` = examples (rows), `n` = features (columns).
- `w` → shape `(n,)`: one weight per feature.
- `X_train[i]` → pulls ONE row, collapses 2D → 1D, shape `(n,)` — now matches `w`.
- Prediction for one example: `np.dot(X_train[i], w) + b`
- This single pattern (row-pull → dot product → add bias) is the backbone of every linear regression prediction you'll write.

## Reshape

- `.reshape(-1, n)` → "make `n` columns, figure out the rows for me."
- `.reshape(3, 2)` and `.reshape(-1, 2)` on a 6-element array give the identical result — `-1` just saves the manual division.