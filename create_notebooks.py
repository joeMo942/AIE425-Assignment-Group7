#!/usr/bin/env python3
"""Script to create empty Jupyter notebooks for each section folder."""

import json
import os

# Define notebook structure
notebooks = [
    ("section1_statistical_analysis", "section1_statistical_analysis.ipynb", "Section 1: Statistical Analysis"),
    ("section2_neighborhood_cf/part1_user_based_cf", "part1_user_based_cf.ipynb", "Part 1: User-Based Collaborative Filtering"),
    ("section2_neighborhood_cf/part2_item_based_cf", "part2_item_based_cf.ipynb", "Part 2: Item-Based Collaborative Filtering"),
    ("section3_clustering_based_cf/part1_user_clustering_avg_ratings", "part1_user_clustering_avg_ratings.ipynb", "Part 1: User Clustering with Average Ratings"),
    ("section3_clustering_based_cf/part2_user_clustering_common_ratings", "part2_user_clustering_common_ratings.ipynb", "Part 2: User Clustering with Common Ratings"),
    ("section3_clustering_based_cf/part3_item_clustering_avg_raters", "part3_item_clustering_avg_raters.ipynb", "Part 3: Item Clustering with Average Raters"),
    ("section3_clustering_based_cf/part4_cold_start_clustering", "part4_cold_start_clustering.ipynb", "Part 4: Cold Start Clustering"),
]

# Template for empty notebook with title
def create_notebook_template(title):
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n\n---\n\n## Overview\n\nThis notebook implements {title.lower()}.\n\n---"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["# Import required libraries\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Set visualization style\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (12, 6)"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["---\n\n## Load Data"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["# Load dataset\n# TODO: Add data loading code here"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

# Create notebooks
for folder, filename, title in notebooks:
    filepath = os.path.join(folder, filename)
    notebook = create_notebook_template(title)
    
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Created: {filepath}")

print("\n🎉 All notebooks created successfully!")
