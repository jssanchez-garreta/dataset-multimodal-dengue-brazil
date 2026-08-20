# ============================================================
# 27_generate_paper_tables.py
#
# PROYECTO:
# Early Detection of Dengue Outbreaks in Brazil
#
# OBJETIVO:
#
# Convertir tablas CSV del EDA a tablas LaTeX
# listas para el artículo.
#
# ENTRADAS:
#
# dataset_overview.csv
# missing_values.csv
# correlation_ranking.csv
# feature_groups.csv
#
# SALIDAS:
#
# PAPER_TABLES/
#
# ├── table_dataset_overview.tex
# ├── table_missing_values.tex
# ├── table_correlations.tex
# └── table_feature_groups.tex
#
# ============================================================

# ==================================================
# IMPORTS
# ==================================================

import os
import pandas as pd

# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL"
)

EDA_DIR = os.path.join(
    PROJECT_DIR,
    "DATASETS_MASTER_MULTIMODAL",
    "EDA_V8"
)

CSV_DIR = os.path.join(
    EDA_DIR,
    "TABLES_CSV"
)

OUTPUT_DIR = os.path.join(
    EDA_DIR,
    "PAPER_TABLES"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==================================================
# LOAD TABLES
# ==================================================

overview = pd.read_csv(
    os.path.join(
        CSV_DIR,
        "dataset_overview.csv"
    )
)

missing = pd.read_csv(
    os.path.join(
        CSV_DIR,
        "missing_values.csv"
    )
)

corr = pd.read_csv(
    os.path.join(
        CSV_DIR,
        "correlation_ranking.csv"
    )
)

feature_groups = pd.read_csv(
    os.path.join(
        CSV_DIR,
        "feature_groups.csv"
    )
)

# ==================================================
# DATASET OVERVIEW
# ==================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "table_dataset_overview.tex"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(

        overview.to_latex(
            index=False,
            escape=False,
            caption=(
                "Overview of "
                "DATASET_MULTIMODAL_V8."
            ),
            label="tab:dataset_overview"
        )

    )

print(
    "table_dataset_overview.tex"
)

# ==================================================
# MISSING VALUES
# ==================================================

top_missing = missing.head(20)

with open(
    os.path.join(
        OUTPUT_DIR,
        "table_missing_values.tex"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(

        top_missing.to_latex(
            index=False,
            float_format="%.2f",
            escape=False,
            caption=(
                "Top 20 variables "
                "with missing values."
            ),
            label="tab:missing_values"
        )

    )

print(
    "table_missing_values.tex"
)

# ==================================================
# CORRELATIONS
# ==================================================

top_corr = corr.head(20)

with open(
    os.path.join(
        OUTPUT_DIR,
        "table_correlations.tex"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(

        top_corr.to_latex(
            index=False,
            float_format="%.4f",
            escape=False,
            caption=(
                "Top 20 variables "
                "correlated with dengue cases."
            ),
            label="tab:correlations"
        )

    )

print(
    "table_correlations.tex"
)

# ==================================================
# FEATURE GROUPS
# ==================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "table_feature_groups.tex"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(

        feature_groups.to_latex(
            index=False,
            escape=False,
            caption=(
                "Feature groups represented "
                "in DATASET_MULTIMODAL_V8."
            ),
            label="tab:feature_groups"
        )

    )

print(
    "table_feature_groups.tex"
)

# ==================================================
# MASTER FILE
# ==================================================

master_tex = f"""
% ==================================================
% DATASET_MULTIMODAL_V8
% PAPER TABLES
% ==================================================

\\input{{table_dataset_overview.tex}}

\\input{{table_feature_groups.tex}}

\\input{{table_missing_values.tex}}

\\input{{table_correlations.tex}}
"""

with open(
    os.path.join(
        OUTPUT_DIR,
        "paper_tables.tex"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(master_tex)

print(
    "paper_tables.tex"
)

# ==================================================
# END
# ==================================================

print("\n===================================")
print("LATEX TABLES GENERATED")
print("===================================")

print(
    f"\nOutput folder:\n{OUTPUT_DIR}"
)

print("\nDONE")