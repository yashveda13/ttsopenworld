1. raw_val_all_onecol.csv -> raw val set of VAST

2. raw_test_all_onecol.csv -> raw test set of VAST

3. raw_train_all_subset_kg_epoch_led_onecol.csv -> 7,406 generated targets with original docs

4. raw_train_all_subset_kg_epoch_onecol.csv -> exactly same with raw_train_all_subset_kg_epoch_led_onecol.csv

5. raw_train_all_generated_kg_zerotrn.csv -> a portion of raw_train_all_subset_kg_epoch_led_onecol.csv with confidence above a predefined threshold, which is used for training the teacher model

4 and 5 are not required for running the code but will be generated during running.