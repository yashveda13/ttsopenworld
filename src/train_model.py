import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import argparse
import json
import os
import gc
import warnings
import utils.preprocessing as pp
import utils.data_helper as dh
from transformers import AdamW
from utils import modeling, evaluation, model_utils
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report

from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def compute_performance(preds, y, trainvaltest, step, args, seed):
    print("preds size:", preds.size())
    print("y size:", y.size())
    preds_np = preds.cpu().numpy()
    preds_np = np.argmax(preds_np, axis=1)
    y_train2_np = y.cpu().numpy()
    results_weighted = precision_recall_fscore_support(
        y_train2_np, preds_np, average="macro"
    )

    print(
        "-------------------------------------------------------------------------------------"
    )
    print(trainvaltest + " classification_report for step: {}".format(step))
    target_names = ["AGAINST", "FAVOR", "NONE"]
    print(
        classification_report(
            y_train2_np, preds_np, target_names=target_names, digits=4
        )
    )
    ###############################################################################################
    ################            Precision, recall, F1 to csv                     ##################
    ###############################################################################################
    results_twoClass = precision_recall_fscore_support(
        y_train2_np, preds_np, average=None
    )
    if args["task_name"] == "semeval":
        results_weighted = 3 * [(results_twoClass[2][0] + results_twoClass[2][1]) / 2]
    else:
        results_weighted = precision_recall_fscore_support(
            y_train2_np, preds_np, average="macro"
        )
    print("results_weighted:", results_weighted)
    result_overall = [results_weighted[0], results_weighted[1], results_weighted[2]]
    result_against = [
        results_twoClass[0][0],
        results_twoClass[1][0],
        results_twoClass[2][0],
    ]
    result_favor = [
        results_twoClass[0][1],
        results_twoClass[1][1],
        results_twoClass[2][1],
    ]
    result_neutral = [
        results_twoClass[0][2],
        results_twoClass[1][2],
        results_twoClass[2][2],
    ]

    print("result_overall:", result_overall)
    print("result_favor:", result_favor)
    print("result_against:", result_against)
    print("result_neutral:", result_neutral)

    result_id = ["train", args["gen"], step, seed, args["dropout"], args["dropoutrest"]]
    result_one_sample = (
        result_id + result_against + result_favor + result_neutral + result_overall
    )
    result_one_sample = [result_one_sample]
    print("result_one_sample:", result_one_sample)

    results_df = pd.DataFrame(result_one_sample)
    print("results_df are:", results_df.head())
    results_df.to_csv(
        "./results_" + trainvaltest + "_df.csv", index=False, mode="a", header=False
    )
    print("./results_" + trainvaltest + "_df.csv save, done!")
    print(
        "----------------------------------------------------------------------------"
    )

    return results_weighted[2], result_one_sample


def ensure_results_folder():
    if not os.path.exists("./results"):
        os.makedirs("./results")


def get_default_config(config_type="bert"):
    if config_type == "bert":
        return {
            "model_select": "Bart",
            "bert_lr": "2e-5",
            "fc_lr": "1e-3",
            "batch_size": "64",
            "total_epochs": "4",
            "max_tok_len": "200",
            "max_tar_len": "10",
            "dropout": "0.",
        }
    else:  # gen config
        return {
            "model_select": "Bart-MNLI",
            "bert_lr": "2e-5",
            "fc_lr": "1e-3",
            "batch_size": "64",
            "total_epochs": "4",
            "max_tok_len": "200",
            "max_tar_len": "10",
            "dropout": "0.",
        }


def run_classifier():
    # Create results folder if it doesn't exist
    ensure_results_folder()

    # Parse arguments first
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_type",
        default="bert",
        help="Type of config to use: bert or gen",
        required=False,
    )
    parser.add_argument(
        "-g",
        "--gen",
        default="0",
        help="Generation number of student model",
        required=False,
    )
    parser.add_argument(
        "-s", "--seed", default="42", help="Random seed", required=False
    )
    parser.add_argument(
        "-d", "--dropout", default="0.1", help="Dropout rate", required=False
    )
    parser.add_argument(
        "-d2",
        "--dropoutrest",
        default="0.1",
        help="Dropout rate for rest generations",
        required=False,
    )
    parser.add_argument(
        "-train",
        "--train_data",
        default="/content/ttsopenworld/data_all/raw_train_all_subset_kg_epoch_led_onecol.csv",
        help="Name of the training data file",
        required=False,
    )
    parser.add_argument(
        "-dev",
        "--dev_data",
        default="/content/ttsopenworld/data_all/raw_val_all_onecol.csv",
        help="Name of the dev data file",
        required=False,
    )
    parser.add_argument(
        "-test",
        "--test_data",
        default="/content/ttsopenworld/data_all/raw_val_all_onecol.csv",
        help="Name of the test data file",
        required=False,
    )
    parser.add_argument(
        "-kg",
        "--kg_data",
        default="/content/ttsopenworld/data_all/raw_train_all_subset_kg_epoch_led_onecol.csv",
        help="Name of the kg test data file",
        required=False,
    )
    parser.add_argument(
        "-clipgrad",
        "--clipgradient",
        type=str,
        default="True",
        help="whether clip gradient when over 2",
        required=False,
    )
    parser.add_argument(
        "-n", "--clip_norm", type=int, default=2, help="clip gradient", required=False
    )
    parser.add_argument(
        "-exc",
        "--exclude",
        default="none",
        help="which target to exclude and test",
        required=False,
    )
    parser.add_argument(
        "-step",
        "--savestep",
        type=int,
        default=1,
        help="Save step interval",
        required=False,
    )
    parser.add_argument(
        "-p", "--percent", type=int, default=1, help="Data percentage", required=False
    )
    parser.add_argument(
        "-es_step",
        "--earlystopping_step",
        type=int,
        default=1,
        help="Early stopping patience",
        required=False,
    )
    parser.add_argument("-z", "--zero_trn", action="store_true", required=False)
    parser.add_argument(
        "-t",
        "--task_name",
        default="semeval",
        help="Name of the stance dataset",
        required=False,
    )

    args = vars(parser.parse_args())

    # Now create header and write to result files
    header = pd.DataFrame(
        [
            f"{'#'*10}{args['exclude']}_seed_{args['seed']}_d0_{args['dropout']}_d1_{args['dropoutrest']}{'#'*10}"
        ]
    )

    result_files = [
        "results/results_training",
        "results/results_validation",
        "results/results_test",
        "results/best_results_validation",
        "results/best_results_test",
        "results/best_loss_results_validation",
        "results/best_loss_results_test",
    ]

    for file in result_files:
        header.to_csv(f"{file}_df.csv", index=False, mode="a", header=False)

    # Get config directly instead of loading from file
    config = get_default_config(args["config_type"])

    sheet_num = 4  # Google sheet number
    num_labels = 3  # Favor, Against and None
    random_seeds = []
    random_seeds.append(int(args["seed"]))

    # create normalization dictionary for preprocessing
    with open("/content/ttsopenworld/src/noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("/content/ttsopenworld/src/emnlp_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split("\t")
            data2[row[0]] = row[1].rstrip()
    norm_dict = {**data1, **data2}

    # Use GPU or not
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    for seed in random_seeds:
        print("current random seed: ", seed)

        log_dir = os.path.join(
            "./tensorboard/tensorboard_train"
            + str(args["percent"])
            + "_d0"
            + str(args["dropout"])
            + "_d1"
            + str(
                args["dropoutrest"] + "_seed" + str(seed) + "_gen" + str(args["gen"])
            ),
            "train",
        )
        train_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join(
            "./tensorboard/tensorboard_train"
            + str(args["percent"])
            + "_d0"
            + str(args["dropout"])
            + "_d1"
            + str(
                args["dropoutrest"] + "_seed" + str(seed) + "_gen" + str(args["gen"])
            ),
            "val",
        )
        val_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join(
            "./tensorboard/tensorboard_train"
            + str(args["percent"])
            + "_d0"
            + str(args["dropout"])
            + "_d1"
            + str(
                args["dropoutrest"] + "_seed" + str(seed) + "_gen" + str(args["gen"])
            ),
            "test",
        )
        test_writer = SummaryWriter(log_dir=log_dir)

        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load and check datasets
        x_train, y_train, x_train_target = pp.clean_all(
            args["train_data"], args["exclude"], args["task_name"], norm_dict
        )
        if not x_train:
            print(f"Error: Training dataset is empty. Cannot proceed with training.")
            return

        x_val, y_val, x_val_target = pp.clean_all(
            args["dev_data"], args["exclude"], args["task_name"], norm_dict
        )
        x_test, y_test, x_test_target = pp.clean_all(
            args["test_data"], args["exclude"], args["task_name"], norm_dict
        )
        x_test_kg, y_test_kg, x_test_target_kg = pp.clean_all(
            args["kg_data"], args["exclude"], args["task_name"], norm_dict
        )

        # Use empty lists if validation/test sets are empty
        x_val = x_val if x_val else []
        y_val = y_val if y_val else []
        x_val_target = x_val_target if x_val_target else []

        x_test = x_test if x_test else []
        y_test = y_test if y_test else []
        x_test_target = x_test_target if x_test_target else []

        x_train_all = [x_train, y_train, x_train_target]
        x_val_all = [x_val, y_val, x_val_target]
        x_test_all = [x_test, y_test, x_test_target]
        x_test_kg_all = [x_test_kg, y_test_kg, x_test_target_kg]
        if int(args["gen"]) >= 1:
            print("Current generation is: ", args["gen"])
            remove_indices = []
            if args["zero_trn"]:
                for i in range(len(x_test_target_kg)):
                    for j in range(len(x_train_target)):
                        if (
                            x_test_target_kg[i] == x_train_target[j]
                            and x_test_kg[i] == x_train[j]
                        ):
                            remove_indices.append(i)
                            break
                for t_id in range(len(x_test_kg_all)):
                    x_test_kg_all[t_id] = [
                        i
                        for j, i in enumerate(x_test_kg_all[t_id])
                        if j not in remove_indices
                    ]
            x_train_all = [a + b for a, b in zip(x_train_all, x_test_kg_all)]
        # Print dataset info safely
        if x_test_all[0] and x_test_all[1] and x_test_all[2]:
            print(
                f"Test sample: {x_test_all[0][0]}, {x_test_all[1][0]}, {x_test_all[2][0]}"
            )
        else:
            print("Warning: Test dataset is empty")

        # Prepare data loaders with validation for empty datasets
        if not x_val_all[0] or not x_test_all[0]:
            print(
                "Warning: Validation or test sets are empty. Training will proceed with only training data."
            )

        loader, gt_label = dh.data_helper_bert(
            x_train_all,
            x_val_all,
            x_test_all,
            x_test_kg_all,
            config["model_select"],
            config,
        )
        trainloader, valloader, testloader, trainloader2, kg_testloader = (
            loader[0],
            loader[1],
            loader[2],
            loader[3],
            loader[4],
        )
        y_train, y_val, y_test, y_train2 = (
            gt_label[0],
            gt_label[1],
            gt_label[2],
            gt_label[3],
        )
        y_val, y_test, y_train2 = (
            y_val.to(device),
            y_test.to(device),
            y_train2.to(device),
        )

        # train setup
        model, optimizer = model_utils.model_setup(
            num_labels,
            config["model_select"],
            device,
            config,
            int(args["gen"]),
            float(args["dropout"]),
            float(args["dropoutrest"]),
        )
        loss_function = nn.CrossEntropyLoss()
        sum_loss = []
        val_f1_average, test_kg = [], []

        # early stopping setup
        early_stopping = EarlyStopping(
            patience=args["earlystopping_step"], verbose=True
        )
        print(100 * "#")
        print(
            f"Early stopping will occur after {args['earlystopping_step']} epochs without improvement"
        )
        print(100 * "#")

        # Initialize result arrays with proper structure
        best_val_result = [[None] * 18]  # Adjust size based on your columns
        best_test_result = [[None] * 18]
        best_val_loss_result = [[None] * 18]
        best_test_loss_result = [[None] * 18]

        # init best val/test results
        best_train_f1macro, best_val_f1macro, best_test_f1macro = -1, -1, -1
        best_val_loss, best_test_loss = float("inf"), float("inf")

        # start training
        print(100 * "#")
        print("clipgradient:", args["clipgradient"] == "True")
        print(100 * "#")
        step = 0
        # Save checkpoint directory
        checkpoint_dir = "./checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initialize validation tracking
        best_val_loss = float("inf")
        patience_counter = 0
        last_val_loss = float("inf")

        for epoch in range(0, int(config["total_epochs"])):
            try:
                print(f"Epoch: {epoch}")
                model.train()
                train_loss = []

                # Training loop
                progress_bar = tqdm(trainloader, desc=f"Epoch {epoch}")
                for b_id, sample_batch in enumerate(progress_bar):
                    model.train()
                    optimizer.zero_grad()
                    dict_batch = model_utils.batch_fn(sample_batch)
                    inputs = {k: v.to(device) for k, v in dict_batch.items()}
                    outputs = model(**inputs)
                    loss = loss_function(outputs, inputs["gt_label"])
                    loss.backward()

                    if args["clipgradient"] == "True":
                        nn.utils.clip_grad_norm_(
                            model.parameters(), int(args["clip_norm"])
                        )

                    optimizer.step()
                    train_loss.append(loss.item())
                    progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

                # Calculate average training loss
                avg_train_loss = sum(train_loss) / len(train_loss)
                print(f"Average training loss: {avg_train_loss:.4f}")

                # Validation phase if validation set exists
                if len(valloader) > 0:
                    model.eval()
                    val_loss = []
                    with torch.no_grad():
                        for val_batch in valloader:
                            dict_batch = model_utils.batch_fn(val_batch)
                            inputs = {k: v.to(device) for k, v in dict_batch.items()}
                            outputs = model(**inputs)
                            val_loss.append(
                                loss_function(outputs, inputs["gt_label"]).item()
                            )

                    avg_val_loss = sum(val_loss) / len(val_loss)
                    print(f"Validation loss: {avg_val_loss:.4f}")

                    # Early stopping check
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= args["earlystopping_step"]:
                            print(
                                f"{100*'!'}\nEarly stopping triggered after {epoch} epochs\n{100*'!'}"
                            )
                            break
                else:
                    # If no validation set, use training loss for early stopping
                    if avg_train_loss < best_val_loss:
                        best_val_loss = avg_train_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= args["earlystopping_step"]:
                            print(
                                f"{100*'!'}\nEarly stopping triggered after {epoch} epochs\n{100*'!'}"
                            )
                            break

            except KeyboardInterrupt:
                print("\nTraining interrupted. Saving checkpoint...")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_train_loss,
                    },
                    f"{checkpoint_dir}/interrupted_checkpoint.pt",
                )
                break

        #########################################################
        if (
            best_val_result[0][0] is None
        ):  # Set default value if not set during training
            best_val_result[0][0] = "best validation"

        results_df = pd.DataFrame(best_val_result)
        results_df.to_csv(
            "./results/results_validation_df.csv", index=False, mode="a", header=False
        )
        print("./results/results_validation_df.csv save, done!")

        results_df.to_csv(
            "./results/best_results_validation_df.csv",
            index=False,
            mode="a",
            header=False,
        )
        print("./results/best_results_validation_df.csv save, done!")

        if best_val_loss_result[0][0] is None:
            best_val_loss_result[0][0] = "best validation"
        results_df = pd.DataFrame(best_val_loss_result)
        results_df.to_csv(
            "./results/best_loss_results_validation_df.csv",
            index=False,
            mode="a",
            header=False,
        )
        print("./results/best_loss_results_validation_df.csv save, done!")
        #########################################################
        best_test_result[0][0] = "best test"
        results_df = pd.DataFrame(best_test_result)
        print("results_df are:", results_df.head())
        results_df.to_csv(
            "./results/results_test_df.csv", index=False, mode="a", header=False
        )
        print("./results/results_test_df.csv save, done!")
        ###
        results_df = pd.DataFrame(best_test_result)
        print("results_df are:", results_df.head())
        results_df.to_csv(
            "./results/best_results_test_df.csv", index=False, mode="a", header=False
        )
        print("./results/best_results_test_df.csv save, done!")
        ###
        best_test_loss_result[0][0] = "best test"
        results_df = pd.DataFrame(best_test_loss_result)
        print("results_df are:", results_df.head())
        results_df.to_csv(
            "./results/best_loss_results_test_df.csv",
            index=False,
            mode="a",
            header=False,
        )
        print("./results/best_loss_results_test_df.csv save, done!")
        #########################################################
        # model that performs best on the dev set is evaluated on the test set
        best_epoch = [
            index for index, v in enumerate(val_f1_average) if v == max(val_f1_average)
        ][-1]

        # update the unlabeled kg file
        if int(args["gen"]) < 1:
            concat_text = pd.read_csv(args["kg_data"], encoding="ISO-8859-1")
            concat_text = concat_text[concat_text["GT Target"] != args["exclude"]]
            concat_text["Stance 1"] = test_kg[best_epoch].tolist()
            concat_text["Stance 1"].replace(
                [0, 1, 2], ["AGAINST", "FAVOR", "NONE"], inplace=True
            )
            concat_text = concat_text.reindex(
                columns=["Tweet", "Target 1", "Stance 1", "seen?", "GT Target"]
            )
            print(100 * "#")
            #             concat_text.to_csv(args['kg_data'][:-4]+'_'+args['exclude']+'_seed{}.csv'.format(seed), index=False)
            #             print(args['kg_data'][:-4]+'_'+args['exclude']+'_seed{}.csv'.format(seed),"save, done!")
            concat_text.to_csv(args["kg_data"], index=False)
            print(args["kg_data"], "save, done!")
            print(100 * "#")

        # Save results
        save_result = {
            "best_against": best_against,
            "best_favor": best_favor,
            "best_result": best_result,
            "best_val_against": best_val_against,
            "best_val_favor": best_val_favor,
            "best_val": best_val,
        }

        # Save as CSV
        results_df = pd.DataFrame(save_result)
        results_df.to_csv("./results/best_results.csv", index=False)
        print("Best results saved to ./results/best_results.csv")

        # Save as JSON
        with open("./results/best_results.json", "w") as f:
            json.dump(save_result, f)
        print("Best results saved to ./results/best_results.json")


if __name__ == "__main__":
    run_classifier()
