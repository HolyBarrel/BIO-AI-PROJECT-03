from util.xgb_feature_selector import FeatureSelectionXGB

# ---------- Example run ---------- 
"""
You can currently use:
file = "1595_poker"
file = "titanic"
file = "breast_cancer_wisconsin_original"
file = "wine_quality_combined"
file = "banana"
"""

if __name__ == "__main__":
    file = "breast_cancer_wisconsin_original"
    
    # Create an instance of FeatureSelectionXGB
    selector = FeatureSelectionXGB(
        df_file_name=file, 
        target_column='target', 
        test_size=0.2,
        ignored_features=['wine_type'], 
        print_per=100, 
        use_gpu=False, 
        show_lookup=True, 
        show_best=True,
        save_to_csv=True, 
        output_location='XGB-Feature-Selection/output'
    )

    # Run the feature selection experiment
    selector.feature_selection_experiment()
