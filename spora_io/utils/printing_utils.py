import shutil
from typing import List

def print_list_of_strings_multicol(lines: List[str], term_width: int | None=None, indent: int=0, pad: int=2) -> None:
    """Print list of strings in multiple columns
    Original: https://gist.github.com/critiqjo/2ca84db26daaeb1715e1
    Adjusted: https://gist.github.com/Nachtalb/8a85c0793b4bea0a102b7414be5888d4

    Args:
        lines (List[str]): List of strings to print
        term_width (int | None): Width of the terminal, if None, it will be determined automatically
        indent (int): Indentation for each line
        pad (int): Padding between columns
    
    Returns:
        None: This function prints the lines directly to the console
    """
    if not term_width:
        size = shutil.get_terminal_size((80, 20))
        term_width = size.columns

    n_lines = len(lines)
    if n_lines == 0:
        return

    col_width = max(len(line) for line in lines)
    n_cols = int((term_width + pad - indent) / (col_width + pad))
    n_cols = min(n_lines, max(1, n_cols))

    col_len = int(n_lines / n_cols) + (0 if n_lines % n_cols == 0 else 1)
    if (n_cols - 1) * col_len >= n_lines:
        n_cols -= 1

    cols = [lines[i * col_len: i * col_len + col_len] for i in range(n_cols)]

    rows = list(zip(*cols))
    rows_missed = zip(*[col[len(rows):] for col in cols[:-1]])
    rows.extend(rows_missed)

    for row in rows:
        print(" " * indent + (" " * pad).join(line.ljust(col_width) for line in row))