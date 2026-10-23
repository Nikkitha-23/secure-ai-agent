"""
tools/science_tool.py — Scientific Computing Tool
--------------------------------------------------
One tool. Five capabilities:
  1. SymPy     → symbolic math (integrals, derivatives, equations, series)
  2. NumPy     → numerical computing (matrices, eigenvalues, FFT)
  3. Pandas    → data analysis (CSV stats, describe, groupby)
  4. SciPy     → statistics (mean, std, correlation, t-test)
  5. Matplotlib → plotting (line, bar, scatter → base64 PNG)

Usage in agent_loop.py:
    from tools.science_tool import run_science_tool, is_science_intent
"""

import logging
import re
import io
import base64
from typing import Optional

logger = logging.getLogger(__name__)


# ── Intent Detection ──────────────────────────────────────────────────────────

SCIENCE_KEYWORDS = {
    # Symbolic math
    "integrate", "integral", "differentiate", "derivative", "d/dx",
    "solve equation", "simplify", "expand", "factor", "limit",
    "series expansion", "taylor series", "laplace", "fourier",
    "matrix", "determinant", "eigenvalue", "eigenvector", "inverse matrix",
    "dot product", "cross product", "transpose",
    # Stats / numerical
    "mean", "median", "mode", "standard deviation", "std dev", "variance",
    "correlation", "covariance", "t-test", "p-value", "regression",
    "histogram", "normal distribution", "probability",
    # Data
    "analyze csv", "analyse csv", "summarize data", "data summary",
    "describe dataset", "group by", "pivot", "filter rows",
    # Plotting
    "plot", "graph", "chart", "visualize", "draw curve",
    "scatter plot", "bar chart", "line graph",
}

SCIENCE_PATTERNS = [
    r"\bintegrate\b", r"\bd/dx\b", r"\bderivative\s+of\b",
    r"\beigenvalue", r"\bmatrix\b", r"\bdeterminant\b",
    r"\bmean\s+of\b", r"\bstd\s+dev\b", r"\bcorrelation\b",
    r"\bplot\s+\w", r"\bgraph\s+of\b", r"\bchart\b",
    r"∫", r"∂", r"∑", r"√",
]


def is_science_intent(query: str) -> bool:
    """Return True if query looks like a scientific/math/data task."""
    q = query.lower()
    for kw in SCIENCE_KEYWORDS:
        if kw in q:
            logger.debug(f"🔬 Science keyword match: '{kw}'")
            return True
    for pat in SCIENCE_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            logger.debug(f"🔬 Science pattern match: '{pat}'")
            return True
    return False


# ── Sub-tool: SymPy ───────────────────────────────────────────────────────────

def run_sympy(query: str) -> str:
    """Symbolic math via SymPy."""
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

        transformations = standard_transformations + (implicit_multiplication_application,)
        x, y, z, t, n = sp.symbols('x y z t n')
        local_dict = {'x': x, 'y': y, 'z': z, 't': t, 'n': n,
                      'pi': sp.pi, 'e': sp.E, 'oo': sp.oo,
                      'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                      'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt}

        q = query.lower()

        # --- Integral ---
        if any(w in q for w in ["integrate", "integral", "∫"]):
            # Extract expression: "integrate x^2" or "integral of sin(x)"
            expr_str = re.sub(r'(integrate|integral\s+of?|∫)\s*', '', q, flags=re.IGNORECASE).strip()
            # Remove "with respect to x" style endings
            expr_str = re.sub(r'\s+(with\s+respect\s+to\s+\w+|dx|dy|dz)$', '', expr_str).strip()
            expr_str = expr_str.replace('^', '**')
            expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
            result = sp.integrate(expr, x)
            return f"∫ {sp.latex(expr)} dx = {sp.latex(result)} + C\n\nSymPy: {result} + C"

        # --- Derivative ---
        if any(w in q for w in ["differentiate", "derivative", "d/dx", "∂"]):
            expr_str = re.sub(r'(differentiate|derivative\s+of?|d/dx)\s*', '', q, flags=re.IGNORECASE).strip()
            expr_str = expr_str.replace('^', '**')
            expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
            result = sp.diff(expr, x)
            return f"d/dx [{sp.latex(expr)}] = {sp.latex(result)}\n\nSymPy: {result}"

        # --- Solve equation ---
        if any(w in q for w in ["solve", "roots of", "find x"]):
            expr_str = re.sub(r'(solve|roots?\s+of|find\s+x(\s+in)?|=\s*0)\s*', '', q, flags=re.IGNORECASE).strip()
            expr_str = expr_str.replace('^', '**')
            expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
            result = sp.solve(expr, x)
            return f"Solutions for {sp.latex(expr)} = 0:\nx = {result}"

        # --- Simplify ---
        if "simplify" in q or "expand" in q or "factor" in q:
            expr_str = re.sub(r'(simplify|expand|factor)\s*', '', q, flags=re.IGNORECASE).strip()
            expr_str = expr_str.replace('^', '**')
            expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
            if "expand" in q:
                result = sp.expand(expr)
            elif "factor" in q:
                result = sp.factor(expr)
            else:
                result = sp.simplify(expr)
            return f"Result: {sp.latex(result)}\n\nSymPy: {result}"

        # --- Matrix / determinant / eigenvalue ---
        if any(w in q for w in ["matrix", "determinant", "eigenvalue", "eigenvector"]):
            # Try to extract a matrix from query: [[1,2],[3,4]]
            mat_match = re.search(r'\[\s*\[.*?\]\s*\]', query)
            if mat_match:
                mat_str = mat_match.group(0)
                mat = sp.Matrix(eval(mat_str))
                results = [f"Matrix:\n{mat}"]
                if "determinant" in q or "det" in q:
                    results.append(f"Determinant: {mat.det()}")
                if "eigenvalue" in q:
                    results.append(f"Eigenvalues: {mat.eigenvals()}")
                if "eigenvector" in q:
                    results.append(f"Eigenvectors: {mat.eigenvects()}")
                if "inverse" in q:
                    try:
                        results.append(f"Inverse:\n{mat.inv()}")
                    except Exception:
                        results.append("Matrix is not invertible (singular)")
                return "\n\n".join(results)
            return "Please provide the matrix in format: [[row1], [row2], ...]"

        # --- Limit ---
        if "limit" in q:
            # "limit of sin(x)/x as x->0"
            match = re.search(r'limit\s+of\s+(.+?)\s+as\s+x\s*->\s*(\S+)', q)
            if match:
                expr_str = match.group(1).replace('^', '**')
                point_str = match.group(2).replace('oo', 'sp.oo').replace('inf', 'sp.oo')
                expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
                point = eval(point_str, {"sp": sp})
                result = sp.limit(expr, x, point)
                return f"lim(x→{point_str}) [{sp.latex(expr)}] = {result}"
            return "Format: 'limit of [expr] as x -> [value]'"

        # Fallback: try to evaluate as expression
        expr_str = query.replace('^', '**')
        expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
        result = sp.simplify(expr)
        return f"Simplified: {sp.latex(result)}\n\nValue: {result}"

    except ImportError:
        return "SymPy not installed. Run: pip install sympy"
    except Exception as e:
        logger.error(f"SymPy error: {e}")
        return f"SymPy could not parse the expression: {e}\nTip: Use 'x' as the variable. Example: 'integrate x^2'"


# ── Sub-tool: NumPy ───────────────────────────────────────────────────────────

def run_numpy(query: str) -> str:
    """Numerical computing via NumPy."""
    try:
        import numpy as np
        q = query.lower()

        # Extract matrix from query
        mat_match = re.search(r'\[\s*\[.*?\]\s*\]', query)

        if mat_match:
            mat = np.array(eval(mat_match.group(0)), dtype=float)
            results = [f"Matrix shape: {mat.shape}\n{mat}"]

            if "eigenvalue" in q or "eigenvector" in q:
                vals, vecs = np.linalg.eig(mat)
                results.append(f"Eigenvalues: {vals}")
                if "eigenvector" in q:
                    results.append(f"Eigenvectors:\n{vecs}")

            if "determinant" in q or "det" in q:
                results.append(f"Determinant: {np.linalg.det(mat):.6f}")

            if "inverse" in q:
                try:
                    results.append(f"Inverse:\n{np.linalg.inv(mat)}")
                except np.linalg.LinAlgError:
                    results.append("Matrix is singular — cannot invert")

            if "transpose" in q:
                results.append(f"Transpose:\n{mat.T}")

            if "rank" in q:
                results.append(f"Rank: {np.linalg.matrix_rank(mat)}")

            return "\n\n".join(results)

        # Array from flat list
        arr_match = re.search(r'\[[\d\s,.-]+\]', query)
        if arr_match:
            arr = np.array(eval(arr_match.group(0)), dtype=float)
            return (
                f"Array: {arr}\n"
                f"Sum: {arr.sum():.4f}\n"
                f"Mean: {arr.mean():.4f}\n"
                f"Std: {arr.std():.4f}\n"
                f"Min: {arr.min():.4f}, Max: {arr.max():.4f}"
            )

        return "Provide a NumPy array or matrix. Example: [[1,2],[3,4]]"

    except ImportError:
        return "NumPy not installed. Run: pip install numpy"
    except Exception as e:
        logger.error(f"NumPy error: {e}")
        return f"NumPy error: {e}"


# ── Sub-tool: Pandas ──────────────────────────────────────────────────────────

def run_pandas(query: str, csv_path: Optional[str] = None) -> str:
    """Data analysis via Pandas."""
    try:
        import pandas as pd
        import os

        # Find CSV file
        if not csv_path:
            upload_dir = "data/uploads"
            if os.path.exists(upload_dir):
                csvs = [f for f in os.listdir(upload_dir) if f.endswith('.csv')]
                if csvs:
                    csv_path = os.path.join(upload_dir, csvs[0])

        if not csv_path or not os.path.exists(csv_path):
            return "No CSV file found. Please upload a CSV file first."

        df = pd.read_csv(csv_path)
        q = query.lower()
        results = [f"File: {os.path.basename(csv_path)} ({len(df)} rows × {len(df.columns)} cols)"]

        if "describe" in q or "summary" in q or "summarize" in q:
            results.append(f"Summary:\n{df.describe().to_string()}")

        if "mean" in q:
            numeric = df.select_dtypes(include='number')
            results.append(f"Mean:\n{numeric.mean().to_string()}")

        if "median" in q:
            numeric = df.select_dtypes(include='number')
            results.append(f"Median:\n{numeric.median().to_string()}")

        if "missing" in q or "null" in q or "na" in q:
            results.append(f"Missing values:\n{df.isnull().sum().to_string()}")

        if "column" in q or "columns" in q:
            results.append(f"Columns: {list(df.columns)}")

        if "head" in q or "first" in q:
            results.append(f"First 5 rows:\n{df.head().to_string()}")

        if "correlation" in q or "corr" in q:
            numeric = df.select_dtypes(include='number')
            results.append(f"Correlation matrix:\n{numeric.corr().to_string()}")

        if len(results) == 1:
            # Default: full describe
            results.append(f"Columns: {list(df.columns)}")
            results.append(f"First 3 rows:\n{df.head(3).to_string()}")
            results.append(f"Summary:\n{df.describe().to_string()}")

        return "\n\n".join(results)

    except ImportError:
        return "Pandas not installed. Run: pip install pandas"
    except Exception as e:
        logger.error(f"Pandas error: {e}")
        return f"Pandas error: {e}"


# ── Sub-tool: Statistics (SciPy / statistics) ─────────────────────────────────

def run_stats(query: str) -> str:
    """Statistical analysis."""
    try:
        import statistics
        q = query.lower()

        # Extract numbers from query
        numbers = re.findall(r'-?\d+\.?\d*', query)
        if not numbers:
            return "Please provide a list of numbers. Example: 'mean of 4 8 15 16 23 42'"

        data = [float(n) for n in numbers]
        results = [f"Data: {data} (n={len(data)})"]

        if "mean" in q or "average" in q:
            results.append(f"Mean: {statistics.mean(data):.4f}")

        if "median" in q:
            results.append(f"Median: {statistics.median(data):.4f}")

        if "mode" in q:
            try:
                results.append(f"Mode: {statistics.mode(data)}")
            except statistics.StatisticsError as e:
                results.append(f"Mode: {e}")

        if "std" in q or "standard deviation" in q or "stdev" in q:
            if len(data) > 1:
                results.append(f"Std Dev (sample): {statistics.stdev(data):.4f}")
                results.append(f"Std Dev (population): {statistics.pstdev(data):.4f}")

        if "variance" in q or "var" in q:
            if len(data) > 1:
                results.append(f"Variance (sample): {statistics.variance(data):.4f}")

        if "min" in q or "max" in q or "range" in q:
            results.append(f"Min: {min(data)}, Max: {max(data)}, Range: {max(data)-min(data):.4f}")

        # Try scipy for advanced stats
        try:
            from scipy import stats as scipy_stats
            if "t-test" in q or "ttest" in q:
                # One-sample t-test against mean=0
                t_stat, p_val = scipy_stats.ttest_1samp(data, 0)
                results.append(f"One-sample t-test (μ=0): t={t_stat:.4f}, p={p_val:.4f}")
            if "normal" in q or "distribution" in q:
                stat, p = scipy_stats.normaltest(data)
                results.append(f"Normality test: stat={stat:.4f}, p={p:.4f}")
                results.append("Data looks " + ("normal" if p > 0.05 else "non-normal") + f" (p={'>' if p>0.05 else '<'}0.05)")
        except ImportError:
            pass

        if len(results) == 1:
            # Default: full summary
            results.append(f"Mean: {statistics.mean(data):.4f}")
            results.append(f"Median: {statistics.median(data):.4f}")
            if len(data) > 1:
                results.append(f"Std Dev: {statistics.stdev(data):.4f}")
            results.append(f"Min: {min(data)}, Max: {max(data)}")

        return "\n".join(results)

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return f"Statistics error: {e}"


# ── Sub-tool: Matplotlib ──────────────────────────────────────────────────────

def run_matplotlib(query: str) -> str:
    """Generate a plot and return base64 PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        import io, base64

        q = query.lower()
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.style.use('seaborn-v0_8-whitegrid')

        plotted = False

        # --- Line/function plot: "plot sin(x)" or "graph of x^2" ---
        func_match = re.search(r'(?:plot|graph\s+of?|draw)\s+(.+?)(?:\s+from\s+|$)', query, re.IGNORECASE)
        if func_match:
            expr_str = func_match.group(1).strip()

            # Range
            range_match = re.search(r'from\s+(-?\d+\.?\d*)\s+to\s+(-?\d+\.?\d*)', query, re.IGNORECASE)
            x_min, x_max = (-10, 10)
            if range_match:
                x_min, x_max = float(range_match.group(1)), float(range_match.group(2))

            x = np.linspace(x_min, x_max, 500)
            safe_locals = {
                'x': x, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'abs': np.abs, 'pi': np.pi, 'e': np.e, 'np': np
            }
            expr_clean = expr_str.replace('^', '**')
            try:
                y = eval(expr_clean, {"__builtins__": {}}, safe_locals)
                ax.plot(x, y, linewidth=2, color='#4C8BF5', label=expr_str)
                ax.axhline(0, color='gray', linewidth=0.5)
                ax.axvline(0, color='gray', linewidth=0.5)
                ax.set_title(f"y = {expr_str}", fontsize=14)
                ax.set_xlabel("x"), ax.set_ylabel("y")
                ax.legend()
                plotted = True
            except Exception as e:
                ax.text(0.5, 0.5, f"Cannot plot: {e}", ha='center', va='center', transform=ax.transAxes)
                plotted = True

        # --- Bar chart from numbers ---
        if not plotted and ("bar" in q or "bar chart" in q):
            numbers = re.findall(r'-?\d+\.?\d*', query)
            if numbers:
                vals = [float(n) for n in numbers]
                labels = [f"Item {i+1}" for i in range(len(vals))]
                ax.bar(labels, vals, color='#4C8BF5', alpha=0.8)
                ax.set_title("Bar Chart", fontsize=14)
                plotted = True

        # --- Scatter from pairs ---
        if not plotted and "scatter" in q:
            numbers = re.findall(r'-?\d+\.?\d*', query)
            if len(numbers) >= 4 and len(numbers) % 2 == 0:
                mid = len(numbers) // 2
                xs = [float(n) for n in numbers[:mid]]
                ys = [float(n) for n in numbers[mid:]]
                ax.scatter(xs, ys, color='#4C8BF5', s=80, alpha=0.8)
                ax.set_title("Scatter Plot", fontsize=14)
                plotted = True

        # --- Histogram ---
        if not plotted and "histogram" in q:
            numbers = re.findall(r'-?\d+\.?\d*', query)
            if numbers:
                vals = [float(n) for n in numbers]
                ax.hist(vals, bins='auto', color='#4C8BF5', alpha=0.8, edgecolor='white')
                ax.set_title("Histogram", fontsize=14)
                plotted = True

        if not plotted:
            plt.close()
            return "Could not generate plot. Example: 'plot sin(x)', 'plot x^2 from -5 to 5'"

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"[PLOT:{img_b64}]"

    except ImportError:
        return "Matplotlib not installed. Run: pip install matplotlib"
    except Exception as e:
        logger.error(f"Matplotlib error: {e}")
        return f"Matplotlib error: {e}"


# ── Router: which sub-tool to call? ──────────────────────────────────────────

def _pick_subtool(query: str) -> str:
    q = query.lower()
    # Plotting → Matplotlib
    if any(w in q for w in ["plot", "graph", "chart", "histogram", "scatter", "visualize", "draw curve"]):
        return "matplotlib"
    # Data/CSV → Pandas
    if any(w in q for w in ["csv", "dataframe", "dataset", "rows", "columns", "describe", "missing"]):
        return "pandas"
    # Pure statistics on numbers (no symbols) → stats
    if any(w in q for w in ["mean", "median", "mode", "std", "variance", "t-test", "p-value", "normal"]):
        # If numbers present → stats; if symbols → sympy
        if re.search(r'\d', query):
            return "stats"
    # Matrices with floats → numpy; symbolic → sympy
    if any(w in q for w in ["matrix", "eigenvalue", "determinant", "dot product"]):
        if re.search(r'\[\s*\[', query):
            return "numpy"
    # Default symbolic math → sympy
    return "sympy"


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_science_tool(query: str, csv_path: Optional[str] = None) -> str:
    """
    Main entry point called from agent_loop.py

    Returns a string result (or [PLOT:base64] for images).
    """
    subtool = _pick_subtool(query)
    logger.info(f"🔬 Science tool → subtool: {subtool} | query: {query[:60]}")

    if subtool == "matplotlib":
        return run_matplotlib(query)
    elif subtool == "pandas":
        return run_pandas(query, csv_path)
    elif subtool == "stats":
        return run_stats(query)
    elif subtool == "numpy":
        return run_numpy(query)
    else:
        return run_sympy(query)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("integrate x^2", "sympy"),
        ("differentiate sin(x)", "sympy"),
        ("mean of 1 2 3 4 5", "stats"),
        ("eigenvalues of [[1,2],[3,4]]", "numpy"),
        ("plot sin(x) from -3 to 3", "matplotlib"),
    ]
    print("\n🔬 Science Tool Self-Test:\n")
    for query, expected in tests:
        subtool = _pick_subtool(query)
        status = "✅" if subtool == expected else "⚠️"
        print(f"{status} [{subtool}] {query}")
        result = run_science_tool(query)
        if result.startswith("[PLOT:"):
            print(f"   → Plot generated ({len(result)} chars b64)\n")
        else:
            print(f"   → {result[:120]}\n")
