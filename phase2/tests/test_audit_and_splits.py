"""
Unit tests for FinFact dataset parsing, label preservation, and split leakage detection.
"""

import os
import unittest
import json
import pandas as pd

class TestDatasetAndSplits(unittest.TestCase):
    def setUp(self):
        self.json_path = "finfact.json"
        self.data_dir = "data"
        self.assertTrue(os.path.exists(self.json_path), "finfact.json must exist in root.")

    def test_dataset_record_count_and_labels(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.assertEqual(len(data), 3369, "FinFact benchmark must contain exactly 3,369 records.")
        labels = set(d.get("label") for d in data)
        self.assertEqual(labels, {"false", "true", "NEI"}, "FinFact labels must strictly be 'false', 'true', or 'NEI'.")

    def test_split_sizes_and_preservation(self):
        train_path = os.path.join(self.data_dir, "train.csv")
        val_path = os.path.join(self.data_dir, "val.csv")
        test_path = os.path.join(self.data_dir, "test.csv")
        
        self.assertTrue(os.path.exists(train_path), "train.csv must exist.")
        self.assertTrue(os.path.exists(val_path), "val.csv must exist.")
        self.assertTrue(os.path.exists(test_path), "test.csv must exist.")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        total_rows = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_rows, 3369, "Split rows must sum to exactly 3,369.")
        self.assertEqual(len(train_df), 2358, "Train split should have 2,358 rows (70%).")
        self.assertEqual(len(val_df), 505, "Val split should have 505 rows (15%).")
        self.assertEqual(len(test_df), 506, "Test split should have 506 rows (15%).")

    def test_grouped_split_zero_url_leakage(self):
        g_train_path = os.path.join(self.data_dir, "grouped_train.csv")
        g_test_path = os.path.join(self.data_dir, "grouped_test.csv")
        
        self.assertTrue(os.path.exists(g_train_path), "grouped_train.csv must exist.")
        self.assertTrue(os.path.exists(g_test_path), "grouped_test.csv must exist.")
        
        g_train = pd.read_csv(g_train_path)
        g_test = pd.read_csv(g_test_path)
        
        train_urls = set(g_train["url"].dropna().unique())
        test_urls = set(g_test["url"].dropna().unique())
        
        overlap = train_urls.intersection(test_urls)
        self.assertEqual(len(overlap), 0, f"Source-grouped split must have zero URL overlap! Found {len(overlap)} overlapping URLs.")

if __name__ == "__main__":
    unittest.main()
