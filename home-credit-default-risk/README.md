# Home Credit Default Risk Dataset

## Download Instructions

This folder contains the dataset for the Home Credit Default Risk project.

### How to Download

1. Visit the Kaggle competition page:
   ```
   https://www.kaggle.com/c/home-credit-default-risk/data
   ```

2. Download the following files and place them in this directory:
   - `application_train.csv` - Main training data
   - `application_test.csv` - Main testing data
   - `bureau.csv` - Credit bureau data
   - `bureau_balance.csv` - Monthly balances of previous credits
   - `credit_card_balance.csv` - Monthly balance snapshots of credit cards
   - `installments_payments.csv` - Repayment history
   - `POS_CASH_balance.csv` - Monthly balance snapshots of POS and cash loans
   - `previous_application.csv` - Previous applications for loans
   - `HomeCredit_columns_description.csv` - Column descriptions

3. After downloading, your directory structure should look like:
   ```
   home-credit-default-risk/
   ├── README.md
   ├── application_train.csv
   ├── application_test.csv
   ├── bureau.csv
   ├── bureau_balance.csv
   ├── credit_card_balance.csv
   ├── installments_payments.csv
   ├── POS_CASH_balance.csv
   ├── previous_application.csv
   ├── HomeCredit_columns_description.csv
   └── sample_submission.csv
   ```

## Note

The CSV files are ignored by git (see `.gitignore`) to avoid committing large data files to the repository.
