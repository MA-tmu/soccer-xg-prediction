import pandas as pd

def count_one(csv_path):
    df = pd.read_csv(csv_path)
    num_scored = (df["label"] == 1).sum()
    num_missed = (df["label"] == 0).sum()
    total = len(df)

    print(f"\n[{csv_path}]")
    print(f"  Total  : {total}")
    print(f"  Scored : {num_scored}")
    print(f"  Missed : {num_missed}")

    if num_scored > 0:
        ratio = num_missed / num_scored
        print(f"  Missed : Scored = {ratio:.4f} : 1")
        return ratio
    else:
        print("  No Scored samples!")
        return None


if __name__ == "__main__":
    train_csv = r"E:\xg-twostream\data\splits\train.csv"
    val_csv   = r"E:\xg-twostream\data\splits\val.csv"
    test_csv  = r"E:\xg-twostream\data\splits\test.csv"

    train_ratio = count_one(train_csv)
    val_ratio   = count_one(val_csv)
    test_ratio  = count_one(test_csv)

    if train_ratio is not None:
        print(f"\nUse this for pos_weight (from train set): {train_ratio:.4f}")
