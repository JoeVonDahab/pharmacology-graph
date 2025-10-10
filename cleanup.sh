#!/bin/bash

# Cleanup script to remove unnecessary files for end users
# This keeps only essential files needed to run the Gradio app

echo "🧹 Pharmacology Graph - Cleanup Script"
echo "======================================"
echo ""
echo "This will remove development/duplicate files not needed to run the app."
echo "Essential files (app.py, CSV data, requirements_app.txt) will be kept."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Removing unnecessary files..."

# Documentation duplicates
rm -f BUGFIX_SUMMARY.md && echo "  ✓ Removed BUGFIX_SUMMARY.md"
rm -f APP_SUMMARY.md && echo "  ✓ Removed APP_SUMMARY.md"
rm -f CHECKLIST.md && echo "  ✓ Removed CHECKLIST.md"

# Duplicate/old scripts
rm -f run_app.sh && echo "  ✓ Removed run_app.sh (use start_app.sh instead)"
rm -f test_search.py && echo "  ✓ Removed test_search.py"

# Old/temporary files (if they exist)
rm -f main.py && echo "  ✓ Removed main.py"
rm -f .python-version && echo "  ✓ Removed .python-version"

# Image outputs (can regenerate from notebook)
rm -f *.png && echo "  ✓ Removed PNG images"

# Pickle files (can regenerate from notebook)
rm -f *.pkl && echo "  ✓ Removed PKL files"

# Intermediate CSV files (not needed for app)
rm -f drug_mechanism_filtered.csv && echo "  ✓ Removed drug_mechanism_filtered.csv"
rm -f drug_warnings.csv && echo "  ✓ Removed drug_warnings.csv"
rm -f edges_drug_protein.csv && echo "  ✓ Removed edges_drug_protein.csv"
rm -f proteins_for_embedding.csv && echo "  ✓ Removed proteins_for_embedding.csv"
rm -f top_50_predicted_drug_targets.csv && echo "  ✓ Removed top_50_predicted_drug_targets.csv (duplicate)"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Remaining essential files:"
ls -lh *.py *.txt *.sh *.md 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
echo ""
echo "Data files:"
ls -lh *.csv *.npy 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
echo ""
echo "Total size:"
du -sh . | awk '{print "  " $1}'
