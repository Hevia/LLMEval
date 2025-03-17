from prepare_clean import run as prepare_clean
from validate_results import run as validate_results
from itertools import product

DATASETS = ["CNN-DailyMail", "Samsum", "Xsum"]
PARTITIONS = ["Control", "Treatment"]

if __name__ == "__main__":
    params = product(DATASETS, PARTITIONS)

    for dx, px in params:
        prepare_clean(dx, px)
    
    print("Data prepared. Validating...")

    for dx, px in params:
        validate_results(dx, px)