# scripts/draw_architecture.py
import subprocess
from pathlib import Path


def generate_tikz_body(temporal_type: str) -> str:
    backbone = r"""
\begin{tikzpicture}[
    font=\small,
    block/.style={rectangle, draw, rounded corners=2pt, minimum height=1.2cm,
                  minimum width=2.4cm, align=center, fill=#1},
    dia/.style={shape=diamond, draw, aspect=1.8, inner sep=1pt,
                minimum height=1.25cm, align=center, fill=#1},
    arrow/.style={->, thick},
    node distance=0.55cm
]
\node[block=blue!15] (input) {Input Sequence\\$(B, T, 3, 224, 224)$};
\node[block=orange!20, right=of input] (backbone) {MobileViT-Small\\Backbone};
\draw[arrow] (input) -- (backbone);
"""

    if temporal_type == "none":
        temporal = r"""
\node[dia=green!15, right=of backbone] (pool) {Mean Pooling\\over Time};
\draw[arrow] (backbone) -- (pool);
"""
        last_node = "pool"
        title = "MobileViT + Mean Pooling"
    elif temporal_type == "bilstm":
        temporal = r"""
\node[block=teal!20, right=of backbone] (lstm1) {BiLSTM\\(640 $\to$ 256)};
\node[block=teal!30, right=of lstm1] (lstm2) {BiLSTM\\(512 $\to$ 256)};
\draw[arrow] (backbone) -- (lstm1);
\draw[arrow] (lstm1) -- (lstm2);
"""
        last_node = "lstm2"
        title = "MobileViT + BiLSTM"
    elif temporal_type == "transformer":
        temporal = r"""
\node[block=purple!15, right=of backbone] (trans) {Temporal Transformer\\(1--2 layers)};
\draw[arrow] (backbone) -- (trans);
"""
        last_node = "trans"
        title = "MobileViT + Temporal Transformer"
    elif temporal_type == "bilstm_transformer":
        temporal = r"""
\node[block=teal!20, right=of backbone] (lstm) {BiLSTM};
\node[block=purple!15, right=of lstm] (trans) {Temporal Transformer};
\draw[arrow] (backbone) -- (lstm);
\draw[arrow] (lstm) -- (trans);
"""
        last_node = "trans"
        title = "MobileViT + BiLSTM + Transformer"
    else:
        raise ValueError(f"Unknown type: {temporal_type}")

    head = rf"""
\node[block=red!15, right=of {last_node}] (head) {{FC + Dropout\\+ Softmax}};
\node[dia=gray!30, right=of head] (output) {{Output\\(4 classes)}};
\draw[arrow] ({last_node}) -- (head);
\draw[arrow] (head) -- (output);
% \node[above=0.5cm of input, font=\bfseries] {{{title}}};
\end{{tikzpicture}}
"""
    return backbone + temporal + head


def generate_and_compile(temporal_type: str, output_dir: str = "figures/architectures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_name = f"mobilevit_{temporal_type}"
    body = generate_tikz_body(temporal_type)

    tex_path = output_dir / f"{tex_name}.tex"
    tex_path.write_text(body, encoding="utf-8")
    print(f"Input TeX: {tex_path}")

    standalone_path = output_dir / f"{tex_name}_standalone.tex"
    standalone_path.write_text(
        "\\documentclass[tikz,border=6pt]{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\usetikzlibrary{positioning,shapes.geometric}\n"
        "\\begin{document}\n"
        f"{body}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )

    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(standalone_path)],
            check=True, capture_output=True,
        )
        pdf_src = output_dir / f"{tex_name}_standalone.pdf"
        pdf_dst = output_dir / f"{tex_name}.pdf"
        if pdf_src.exists():
            pdf_src.replace(pdf_dst)
            print(f"PDF created: {pdf_dst}")
        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", str(pdf_dst), str(output_dir / tex_name)],
            check=True, capture_output=True,
        )
        png_file = output_dir / f"{tex_name}-1.png"
        final_png = output_dir / f"{tex_name}.png"
        if png_file.exists():
            png_file.replace(final_png)
            print(f"PNG created: {final_png}")
    except subprocess.CalledProcessError as e:
        log = (e.stdout or e.stderr or b"").decode("utf-8", errors="replace")
        print(f"Compilation failed for {temporal_type}.")
        print("\n".join(log.strip().splitlines()[-25:]))
    except FileNotFoundError:
        print("pdflatex or pdftoppm not found. Only .tex file was created.")


if __name__ == "__main__":
    for m in ["none", "bilstm", "transformer", "bilstm_transformer"]:
        print(f"\n--- Generating {m} ---")
        generate_and_compile(m)