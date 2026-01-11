"""
Custom Transformers for Home Credit Default Risk Feature Engineering
"""

import numpy as np
import pandas as pd
import gc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class NullOutlierFixer(BaseEstimator, TransformerMixin):
    """
    Transformer để xử lý null và outliers trong Application data
    """
    def __init__(self):
        self.mode_values = {}
        self.median_values = {}
    
    def fit(self, X, y=None):
        # Lưu mode và median từ training data
        self.mode_values['CNT_FAM_MEMBERS'] = X['CNT_FAM_MEMBERS'].mode()[0]
        self.median_values['DAYS_EMPLOYED'] = X[X['DAYS_EMPLOYED'] != 365243]['DAYS_EMPLOYED'].median()
        self.n_features_in_ = X.shape[1]  # Mark as fitted
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Xử lý categorical
        df['NAME_FAMILY_STATUS'].fillna('Data_Not_Available', inplace=True)
        df['NAME_HOUSING_TYPE'].fillna('Data_Not_Available', inplace=True)
        df['FLAG_MOBIL'].fillna('Data_Not_Available', inplace=True)
        df['FLAG_EMP_PHONE'].fillna('Data_Not_Available', inplace=True)
        df['FLAG_CONT_MOBILE'].fillna('Data_Not_Available', inplace=True)
        df['FLAG_EMAIL'].fillna('Data_Not_Available', inplace=True)
        df['OCCUPATION_TYPE'].fillna('Data_Not_Available', inplace=True)
        df['CNT_FAM_MEMBERS'].fillna(self.mode_values['CNT_FAM_MEMBERS'], inplace=True)
        df.loc[df['CODE_GENDER'] == 'XNA', 'CODE_GENDER'] = 'M'
        df['NAME_TYPE_SUITE'].fillna('Unaccompanied', inplace=True)
        df.loc[df['NAME_FAMILY_STATUS'] == 'Unknown', 'NAME_FAMILY_STATUS'] = 'Married'
        
        # Xử lý numerical
        df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
        df['DAYS_EMPLOYED'].fillna(self.median_values['DAYS_EMPLOYED'], inplace=True)
        df['AMT_ANNUITY'].fillna(0, inplace=True)
        df['AMT_GOODS_PRICE'].fillna(0, inplace=True)
        df['EXT_SOURCE_1'].fillna(0, inplace=True)
        df['EXT_SOURCE_2'].fillna(0, inplace=True)
        df['EXT_SOURCE_3'].fillna(0, inplace=True)
        
        return df


class ApplicationFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer để tạo features cho Application data
    """
    def __init__(self):
        self.categorical_cols = None
        self.encoder = None
        self.feature_names_out = None
        
    def fit(self, X, y=None):
        df = X.copy()
        
        # A. Các Tỷ Lệ Tài Chính
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['CREDIT_ANNUITY_PERCENT'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
        df['FAMILY_CNT_INCOME_PERCENT'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
        df['CHILDREN_CNT_INCOME_PERCENT'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)
        
        # B. Các Features Về Thời Gian
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
        df['BIRTH_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)
        df['EMPLOYED_REGISTRATION_PERCENT'] = df['DAYS_EMPLOYED'] / (df['DAYS_REGISTRATION'] + 1)
        df['BIRTH_REGISTRATION_PERCENT'] = df['DAYS_BIRTH'] / (df['DAYS_REGISTRATION'] + 1)
        df['ID_REGISTRATION_DIFF'] = df['DAYS_ID_PUBLISH'] - df['DAYS_REGISTRATION']
        df['ANNUITY_LENGTH_EMPLOYED_PERCENT'] = df['CREDIT_TERM'] / (df['DAYS_EMPLOYED'] + 1)
        
        # C. Features Về Tuổi và Thời Gian Hoàn Thành Khoản Vay
        df['AGE_LOAN_FINISH'] = df['DAYS_BIRTH'] * (-1.0/365) + (df['AMT_CREDIT']/(df['AMT_ANNUITY'] + 1)) * (1.0/12)
        
        # D. Features Về Xe và Điện Thoại
        df['CAR_AGE_EMP_PERCENT'] = df['OWN_CAR_AGE'] / (df['DAYS_EMPLOYED'] + 1)
        df['CAR_AGE_BIRTH_PERCENT'] = df['OWN_CAR_AGE'] / (df['DAYS_BIRTH'] + 1)
        df['PHONE_CHANGE_EMP_PERCENT'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_EMPLOYED'] + 1)
        df['PHONE_CHANGE_BIRTH_PERCENT'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_BIRTH'] + 1)
        
        # E. Features Dựa Trên Median Income Theo Nhóm
        df['MEDIAN_INCOME_CONTRACT_TYPE'] = df.groupby('NAME_CONTRACT_TYPE')['AMT_INCOME_TOTAL'].transform('median')
        df['MEDIAN_INCOME_SUITE_TYPE'] = df.groupby('NAME_TYPE_SUITE')['AMT_INCOME_TOTAL'].transform('median')
        df['MEDIAN_INCOME_HOUSING_TYPE'] = df.groupby('NAME_HOUSING_TYPE')['AMT_INCOME_TOTAL'].transform('median')
        df['MEDIAN_INCOME_ORG_TYPE'] = df.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL'].transform('median')
        df['MEDIAN_INCOME_OCCU_TYPE'] = df.groupby('OCCUPATION_TYPE')['AMT_INCOME_TOTAL'].transform('median')
        df['MEDIAN_INCOME_EDU_TYPE'] = df.groupby('NAME_EDUCATION_TYPE')['AMT_INCOME_TOTAL'].transform('median')
        
        # F. Tỷ Lệ Income So Với Median
        df['ORG_TYPE_INCOME_PERCENT'] = df['MEDIAN_INCOME_ORG_TYPE'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['OCCU_TYPE_INCOME_PERCENT'] = df['MEDIAN_INCOME_OCCU_TYPE'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['EDU_TYPE_INCOME_PERCENT'] = df['MEDIAN_INCOME_EDU_TYPE'] / (df['AMT_INCOME_TOTAL'] + 1)
        
        # G. Loại bỏ các cột FLAG_DOCUMENT
        doc_columns = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
        df.drop(columns=doc_columns, inplace=True)
        
        # H. Identify categorical columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Fit OneHotEncoder
        if len(self.categorical_cols) > 0:
            self.encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(df[self.categorical_cols])
            self.feature_names_out = self.encoder.get_feature_names_out(self.categorical_cols)
        
        self.n_features_in_ = X.shape[1]  # Mark as fitted
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # A. Các Tỷ Lệ Tài Chính
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['CREDIT_ANNUITY_PERCENT'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
        df['FAMILY_CNT_INCOME_PERCENT'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
        df['CHILDREN_CNT_INCOME_PERCENT'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)
        
        # B. Các Features Về Thời Gian
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
        df['BIRTH_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)
        df['EMPLOYED_REGISTRATION_PERCENT'] = df['DAYS_EMPLOYED'] / (df['DAYS_REGISTRATION'] + 1)
        df['BIRTH_REGISTRATION_PERCENT'] = df['DAYS_BIRTH'] / (df['DAYS_REGISTRATION'] + 1)
        df['ID_REGISTRATION_DIFF'] = df['DAYS_ID_PUBLISH'] - df['DAYS_REGISTRATION']
        df['ANNUITY_LENGTH_EMPLOYED_PERCENT'] = df['CREDIT_TERM'] / (df['DAYS_EMPLOYED'] + 1)
        
        # C. Features Về Tuổi và Thời Gian Hoàn Thành Khoản Vay
        df['AGE_LOAN_FINISH'] = df['DAYS_BIRTH'] * (-1.0/365) + (df['AMT_CREDIT']/(df['AMT_ANNUITY'] + 1)) * (1.0/12)
        
        # D. Features Về Xe và Điện Thoại
        df['CAR_AGE_EMP_PERCENT'] = df['OWN_CAR_AGE'] / (df['DAYS_EMPLOYED'] + 1)
        df['CAR_AGE_BIRTH_PERCENT'] = df['OWN_CAR_AGE'] / (df['DAYS_BIRTH'] + 1)
        df['PHONE_CHANGE_EMP_PERCENT'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_EMPLOYED'] + 1)
        df['PHONE_CHANGE_BIRTH_PERCENT'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_BIRTH'] + 1)
        
        # E. Features Dựa Trên Median Income Theo Nhóm
        df['MEDIAN_INCOME_CONTRACT_TYPE'] = df.groupby('NAME_CONTRACT_TYPE')['AMT_INCOME_TOTAL'].transform('median')
        df['MEDIAN_INCOME_SUITE_TYPE'] = df.groupby('NAME_TYPE_SUITE')['AMT_INCOME_TOTAL'].transform('median')
        df['MEDIAN_INCOME_HOUSING_TYPE'] = df.groupby('NAME_HOUSING_TYPE')['AMT_INCOME_TOTAL'].transform('median')
        df['MEDIAN_INCOME_ORG_TYPE'] = df.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL'].transform('median')
        df['MEDIAN_INCOME_OCCU_TYPE'] = df.groupby('OCCUPATION_TYPE')['AMT_INCOME_TOTAL'].transform('median')
        df['MEDIAN_INCOME_EDU_TYPE'] = df.groupby('NAME_EDUCATION_TYPE')['AMT_INCOME_TOTAL'].transform('median')
        
        # F. Tỷ Lệ Income So Với Median
        df['ORG_TYPE_INCOME_PERCENT'] = df['MEDIAN_INCOME_ORG_TYPE'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['OCCU_TYPE_INCOME_PERCENT'] = df['MEDIAN_INCOME_OCCU_TYPE'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['EDU_TYPE_INCOME_PERCENT'] = df['MEDIAN_INCOME_EDU_TYPE'] / (df['AMT_INCOME_TOTAL'] + 1)
        
        # G. Loại bỏ các cột FLAG_DOCUMENT
        doc_columns = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
        df.drop(columns=doc_columns, inplace=True)
        
        # H. One-hot encoding with OneHotEncoder
        if len(self.categorical_cols) > 0:
            # Encode categorical columns
            encoded = self.encoder.transform(df[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=self.feature_names_out, index=df.index)
            
            # Drop original categorical columns and add encoded ones
            df = df.drop(columns=self.categorical_cols)
            df = pd.concat([df, encoded_df], axis=1)
        
        # Thay thế inf và NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        return df


class BureauFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer để xử lý Bureau và Bureau Balance data
    """
    def __init__(self, bureau_path, bureau_balance_path):
        self.bureau_path = bureau_path
        self.bureau_balance_path = bureau_balance_path
        self.bureau_features = None
        self.encoder = None
        self.categorical_cols = None
        self.feature_names_out = None
        
    def fit(self, X, y=None):
        # Load bureau data
        bureau = pd.read_csv(self.bureau_path)
        bureau_balance = pd.read_csv(self.bureau_balance_path)
        
        # Mark as fitted
        self.n_features_in_ = X.shape[1]
        
        # Identify categorical columns before transformation
        self.categorical_cols = bureau.select_dtypes(include=['object']).columns.tolist()
        if 'SK_ID_CURR' in self.categorical_cols:
            self.categorical_cols.remove('SK_ID_CURR')
        if 'SK_ID_BUREAU' in self.categorical_cols:
            self.categorical_cols.remove('SK_ID_BUREAU')
        
        # Fit encoder if there are categorical columns
        if len(self.categorical_cols) > 0:
            self.encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(bureau[self.categorical_cols])
            self.feature_names_out = self.encoder.get_feature_names_out(self.categorical_cols)
        
        # Feature Engineering Giai Đoạn 1
        bureau = self._fe_bureau_stage1(bureau)
        
        # Feature Engineering Giai Đoạn 2: Aggregation
        self.bureau_features = self._fe_bureau_stage2(bureau, bureau_balance)
        
        # Cleanup memory
        del bureau, bureau_balance
        gc.collect()
        
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Merge bureau features
        df = df.merge(self.bureau_features, on='SK_ID_CURR', how='left')
        df.fillna(0, inplace=True)
        
        return df
    
    def _fe_bureau_stage1(self, bureau):
        """Feature Engineering Bureau - Giai Đoạn 1"""
        df = bureau.copy()
        
        # Features Về Thời Gian Tín Dụng
        df['CREDIT_DURATION'] = -df['DAYS_CREDIT'] + df['DAYS_CREDIT_ENDDATE']
        df['ENDDATE_DIFF'] = df['DAYS_CREDIT_ENDDATE'] - df['DAYS_ENDDATE_FACT']
        df['UPDATE_DIFF'] = df['DAYS_CREDIT_ENDDATE'] - df['DAYS_CREDIT_UPDATE']
        
        # Features Về Nợ và Tỷ Lệ
        df['DEBT_PERCENTAGE'] = df['AMT_CREDIT_SUM'] / (df['AMT_CREDIT_SUM_DEBT'] + 1)
        df['DEBT_CREDIT_DIFF'] = df['AMT_CREDIT_SUM'] - df['AMT_CREDIT_SUM_DEBT']
        df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT_SUM'] / (df['AMT_ANNUITY'] + 1)
        df['DEBT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT_SUM_DEBT'] / (df['AMT_ANNUITY'] + 1)
        df['CREDIT_OVERDUE_DIFF'] = df['AMT_CREDIT_SUM'] - df['AMT_CREDIT_SUM_OVERDUE']
        
        # Features Về Lịch Sử Vay
        df['CUSTOMER_LOAN_COUNT'] = df.groupby('SK_ID_CURR')['SK_ID_BUREAU'].transform('count')
        df['CUSTOMER_CREDIT_TYPES'] = df.groupby('SK_ID_CURR')['CREDIT_TYPE'].transform('nunique')
        df['AVG_LOAN_TYPE'] = df['CUSTOMER_LOAN_COUNT'] / (df['CUSTOMER_CREDIT_TYPES'] + 1)
        
        # Credit Type Code
        credit_type_mapping = {'Closed': 0, 'Active': 1}
        df['CREDIT_TYPE_CODE'] = df['CREDIT_ACTIVE'].map(credit_type_mapping).fillna(2)
        
        # Tổng Hợp Credit và Debt
        df['TOTAL_CREDIT_SUM'] = df.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].transform('sum')
        df['TOTAL_DEBT_SUM'] = df.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].transform('sum')
        df['CREDIT_DEBT_RATIO'] = df['TOTAL_CREDIT_SUM'] / (df['TOTAL_DEBT_SUM'] + 1)
        
        # One-hot encoding with OneHotEncoder
        if self.encoder is not None and len(self.categorical_cols) > 0:
            encoded = self.encoder.transform(df[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=self.feature_names_out, index=df.index)
            df = df.drop(columns=self.categorical_cols)
            df = pd.concat([df, encoded_df], axis=1)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def _fe_bureau_stage2(self, bureau, bureau_balance):
        """Feature Engineering Bureau - Giai Đoạn 2: Aggregation"""
        # Aggregation Bureau Balance
        bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
            'MONTHS_BALANCE': ['min', 'max', 'mean', 'size']
        })
        bb_agg.columns = ['BB_' + '_'.join(col).upper() for col in bb_agg.columns]
        bb_agg.reset_index(inplace=True)
        
        # Merge Bureau với Bureau Balance
        bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
        
        # Cleanup bureau_balance và bb_agg
        del bureau_balance, bb_agg
        gc.collect()
        
        # Numerical aggregation
        num_agg = {
            'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean', 'sum'],
            'DAYS_CREDIT': ['mean', 'var'],
            'DAYS_CREDIT_UPDATE': ['mean', 'min'],
            'CREDIT_DAY_OVERDUE': ['mean', 'min'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'BB_MONTHS_BALANCE_SIZE': ['mean', 'sum'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
            'AMT_ANNUITY': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM': ['mean', 'sum', 'max']
        }
        
        bureau_agg = bureau.groupby('SK_ID_CURR').agg(num_agg)
        bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]
        bureau_agg.reset_index(inplace=True)
        
        # Active và Closed credits
        active_cols = [col for col in bureau.columns if 'CREDIT_ACTIVE_Active' in col]
        if len(active_cols) > 0:
            active = bureau[bureau[active_cols[0]] == 1]
            active_agg = active.groupby('SK_ID_CURR').agg(num_agg)
            active_agg.columns = ['A_BUREAU_' + '_'.join(col).upper() for col in active_agg.columns]
            active_agg.reset_index(inplace=True)
            bureau_agg = bureau_agg.merge(active_agg, on='SK_ID_CURR', how='left')
        
        closed_cols = [col for col in bureau.columns if 'CREDIT_ACTIVE_Closed' in col]
        if len(closed_cols) > 0:
            closed = bureau[bureau[closed_cols[0]] == 1]
            closed_agg = closed.groupby('SK_ID_CURR').agg(num_agg)
            closed_agg.columns = ['C_BUREAU_' + '_'.join(col).upper() for col in closed_agg.columns]
            closed_agg.reset_index(inplace=True)
            bureau_agg = bureau_agg.merge(closed_agg, on='SK_ID_CURR', how='left')
            
            # Cleanup
            del closed, closed_agg
            gc.collect()
        
        # Cleanup active data
        if 'active' in locals():
            del active, active_agg
            gc.collect()
        
        bureau_agg.fillna(0, inplace=True)
        
        return bureau_agg


class PreviousApplicationFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer để xử lý Previous Application data
    """
    def __init__(self, previous_app_path):
        self.previous_app_path = previous_app_path
        self.prev_features = None
        self.encoder = None
        self.categorical_cols = None
        self.feature_names_out = None
        
    def fit(self, X, y=None):
        # Load previous application
        prev_app = pd.read_csv(self.previous_app_path)
        
        # Mark as fitted
        self.n_features_in_ = X.shape[1]
        
        # Identify categorical columns before transformation
        self.categorical_cols = prev_app.select_dtypes(include=['object']).columns.tolist()
        if 'SK_ID_CURR' in self.categorical_cols:
            self.categorical_cols.remove('SK_ID_CURR')
        if 'SK_ID_PREV' in self.categorical_cols:
            self.categorical_cols.remove('SK_ID_PREV')
        
        # Xử lý outliers
        prev_app.loc[prev_app['DAYS_FIRST_DRAWING'] == 365243, 'DAYS_FIRST_DRAWING'] = np.nan
        prev_app.loc[prev_app['DAYS_FIRST_DUE'] == 365243, 'DAYS_FIRST_DUE'] = np.nan
        prev_app.loc[prev_app['DAYS_LAST_DUE_1ST_VERSION'] == 365243, 'DAYS_LAST_DUE_1ST_VERSION'] = np.nan
        prev_app.loc[prev_app['DAYS_LAST_DUE'] == 365243, 'DAYS_LAST_DUE'] = np.nan
        prev_app.loc[prev_app['DAYS_TERMINATION'] == 365243, 'DAYS_TERMINATION'] = np.nan
        
        # Feature Engineering
        prev_app['APPLICATION_CREDIT_DIFF'] = prev_app['AMT_APPLICATION'] - prev_app['AMT_CREDIT']
        prev_app['APPLICATION_CREDIT_RATIO'] = prev_app['AMT_APPLICATION'] / (prev_app['AMT_CREDIT'] + 1)
        prev_app['CREDIT_TO_ANNUITY_RATIO'] = prev_app['AMT_CREDIT'] / (prev_app['AMT_ANNUITY'] + 1)
        prev_app['DOWN_PAYMENT_TO_CREDIT'] = prev_app['AMT_DOWN_PAYMENT'] / (prev_app['AMT_CREDIT'] + 1)
        
        total_payment = prev_app['AMT_ANNUITY'] * prev_app['CNT_PAYMENT']
        prev_app['SIMPLE_INTERESTS'] = (total_payment / (prev_app['AMT_CREDIT'] + 1) - 1) / (prev_app['CNT_PAYMENT'] + 1)
        
        prev_app['DAYS_LAST_DUE_DIFF'] = prev_app['DAYS_LAST_DUE_1ST_VERSION'] - prev_app['DAYS_LAST_DUE']
        
        # Fit encoder if there are categorical columns
        if len(self.categorical_cols) > 0:
            self.encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(prev_app[self.categorical_cols])
            self.feature_names_out = self.encoder.get_feature_names_out(self.categorical_cols)
        
        # One-hot encoding
        if self.encoder is not None and len(self.categorical_cols) > 0:
            encoded = self.encoder.transform(prev_app[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=self.feature_names_out, index=prev_app.index)
            prev_app = prev_app.drop(columns=self.categorical_cols)
            prev_app = pd.concat([prev_app, encoded_df], axis=1)
        
        prev_app.replace([np.inf, -np.inf], np.nan, inplace=True)
        prev_app.fillna(0, inplace=True)
        
        # Aggregation
        num_agg = {
            'AMT_ANNUITY': ['max', 'mean'],
            'AMT_APPLICATION': ['max', 'mean'],
            'AMT_CREDIT': ['max', 'mean'],
            'AMT_DOWN_PAYMENT': ['max', 'mean'],
            'AMT_GOODS_PRICE': ['mean', 'sum'],
            'HOUR_APPR_PROCESS_START': ['max', 'mean'],
            'RATE_DOWN_PAYMENT': ['max', 'mean'],
            'RATE_INTEREST_PRIMARY': ['max', 'mean'],
            'RATE_INTEREST_PRIVILEGED': ['max', 'mean'],
            'DAYS_DECISION': ['max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
            'APPLICATION_CREDIT_RATIO': ['max', 'mean'],
            'DOWN_PAYMENT_TO_CREDIT': ['max', 'mean'],
            'DAYS_LAST_DUE_DIFF': ['max', 'mean']
        }
        
        prev_agg = prev_app.groupby('SK_ID_CURR').agg(num_agg)
        prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
        prev_agg.reset_index(inplace=True)
        
        # Approved/Refused
        approved_col = [col for col in prev_app.columns if 'NAME_CONTRACT_STATUS_Approved' in col]
        if len(approved_col) > 0:
            approved = prev_app[prev_app[approved_col[0]] == 1]
            approved_agg = approved.groupby('SK_ID_CURR').agg(num_agg)
            approved_agg.columns = ['CS_APP_' + '_'.join(col).upper() for col in approved_agg.columns]
            approved_agg.reset_index(inplace=True)
            prev_agg = prev_agg.merge(approved_agg, on='SK_ID_CURR', how='left')
        
        refused_col = [col for col in prev_app.columns if 'NAME_CONTRACT_STATUS_Refused' in col]
        if len(refused_col) > 0:
            refused = prev_app[prev_app[refused_col[0]] == 1]
            refused_agg = refused.groupby('SK_ID_CURR').agg(num_agg)
            refused_agg.columns = ['CS_REF_' + '_'.join(col).upper() for col in refused_agg.columns]
            refused_agg.reset_index(inplace=True)
            prev_agg = prev_agg.merge(refused_agg, on='SK_ID_CURR', how='left')
            
            # Cleanup
            del refused, refused_agg
            gc.collect()
        
        # Cleanup approved data
        if 'approved' in locals():
            del approved, approved_agg
            gc.collect()
        
        prev_agg.fillna(0, inplace=True)
        self.prev_features = prev_agg
        
        # Cleanup prev_app
        del prev_app
        gc.collect()
        
        return self
    
    def transform(self, X):
        df = X.copy()
        df = df.merge(self.prev_features, on='SK_ID_CURR', how='left')
        df.fillna(0, inplace=True)
        return df


class POSCashFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer để xử lý POS Cash Balance data
    """
    def __init__(self, pos_cash_path):
        self.pos_cash_path = pos_cash_path
        self.pos_features = None
        self.encoder = None
        self.categorical_cols = None
        self.feature_names_out = None
        
    def fit(self, X, y=None):
        pos = pd.read_csv(self.pos_cash_path)
        
        # Mark as fitted
        self.n_features_in_ = X.shape[1]
        
        # Identify categorical columns
        self.categorical_cols = pos.select_dtypes(include=['object']).columns.tolist()
        if 'SK_ID_CURR' in self.categorical_cols:
            self.categorical_cols.remove('SK_ID_CURR')
        if 'SK_ID_PREV' in self.categorical_cols:
            self.categorical_cols.remove('SK_ID_PREV')
        
        # Feature Engineering
        pos['LATE_PAYMENT'] = (pos['SK_DPD'] > 0).astype(int)
        
        # Fit encoder if there are categorical columns
        if len(self.categorical_cols) > 0:
            self.encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(pos[self.categorical_cols])
            self.feature_names_out = self.encoder.get_feature_names_out(self.categorical_cols)
        
        # One-hot encoding
        if self.encoder is not None and len(self.categorical_cols) > 0:
            encoded = self.encoder.transform(pos[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=self.feature_names_out, index=pos.index)
            pos = pos.drop(columns=self.categorical_cols)
            pos = pd.concat([pos, encoded_df], axis=1)
        
        pos.replace([np.inf, -np.inf], np.nan, inplace=True)
        pos.fillna(0, inplace=True)
        
        # Aggregation
        num_agg = {
            'SK_DPD_DEF': ['max', 'mean', 'min'],
            'SK_DPD': ['max', 'mean', 'min'],
            'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'CNT_INSTALMENT': ['max', 'size'],
            'CNT_INSTALMENT_FUTURE': ['max', 'size', 'sum'],
            'LATE_PAYMENT': ['mean', 'sum']
        }
        
        pos_agg = pos.groupby('SK_ID_CURR').agg(num_agg)
        pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
        pos_agg.reset_index(inplace=True)
        pos_agg.fillna(0, inplace=True)
        
        self.pos_features = pos_agg
        
        # Cleanup pos
        del pos
        gc.collect()
        
        return self
    
    def transform(self, X):
        df = X.copy()
        df = df.merge(self.pos_features, on='SK_ID_CURR', how='left')
        df.fillna(0, inplace=True)
        return df


class InstallmentsPaymentsFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer để xử lý Installments Payments data
    """
    def __init__(self, installments_path):
        self.installments_path = installments_path
        self.ins_features = None
        
    def fit(self, X, y=None):
        ins = pd.read_csv(self.installments_path)
        
        # Mark as fitted
        self.n_features_in_ = X.shape[1]
        
        # Group payments
        ins_grouped = ins.groupby(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])['AMT_PAYMENT'].sum().reset_index()
        ins_grouped.rename(columns={'AMT_PAYMENT': 'AMT_PAYMENT_GROUPED'}, inplace=True)
        
        ins = ins.merge(ins_grouped, on=['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], how='left')
        
        # Features
        ins['PAYMENT_DIFFERENCE'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT_GROUPED']
        ins['PAYMENT_RATIO'] = ins['AMT_INSTALMENT'] / (ins['AMT_PAYMENT_GROUPED'] + 1)
        ins['PAID_OVER_AMOUNT'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
        ins['PAID_OVER'] = (ins['PAID_OVER_AMOUNT'] > 0).astype(int)
        
        # DPD features
        ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins.loc[ins['DPD'] <= 0, 'DPD'] = 0
        
        ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins.loc[ins['DBD'] <= 0, 'DBD'] = 0
        ins['LATE_PAYMENT'] = (ins['DBD'] > 0).astype(int)
        
        ins['INSTALMENT_PAYMENT_RATIO'] = ins['AMT_PAYMENT'] / (ins['AMT_INSTALMENT'] + 1)
        ins['LATE_PAYMENT_RATIO'] = ins['INSTALMENT_PAYMENT_RATIO'] * ins['LATE_PAYMENT']
        ins['SIGNIFICANT_LATE_PAYMENT'] = (ins['LATE_PAYMENT_RATIO'] > 0.05).astype(int)
        
        # DPD Buckets
        ins['DPD_7'] = (ins['DPD'] >= 7).astype(int)
        ins['DPD_15'] = (ins['DPD'] >= 15).astype(int)
        ins['DPD_30'] = (ins['DPD'] >= 30).astype(int)
        ins['DPD_60'] = (ins['DPD'] >= 60).astype(int)
        ins['DPD_90'] = (ins['DPD'] >= 90).astype(int)
        ins['DPD_180'] = (ins['DPD'] >= 180).astype(int)
        ins['DPD_WOF'] = (ins['DPD'] >= 720).astype(int)
        
        ins.replace([np.inf, -np.inf], np.nan, inplace=True)
        ins.fillna(0, inplace=True)
        
        # Aggregation
        num_agg = {
            'LATE_PAYMENT': ['max', 'mean', 'min'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'NUM_INSTALMENT_NUMBER': ['max'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'PAYMENT_DIFFERENCE': ['max', 'mean', 'min', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
            'PAID_OVER_AMOUNT': ['max', 'mean', 'min'],
            'DPD': ['max', 'mean', 'sum'],
            'DPD_7': ['mean', 'sum'],
            'DPD_15': ['mean', 'sum'],
            'DPD_30': ['mean', 'sum'],
            'DPD_60': ['mean', 'sum'],
            'DPD_90': ['mean', 'sum'],
            'DPD_180': ['mean', 'sum'],
            'DPD_WOF': ['mean', 'sum']
        }
        
        ins_agg = ins.groupby('SK_ID_CURR').agg(num_agg)
        ins_agg.columns = ['INS_' + '_'.join(col).upper() for col in ins_agg.columns]
        ins_agg.reset_index(inplace=True)
        ins_agg.fillna(0, inplace=True)
        
        self.ins_features = ins_agg
        
        # Cleanup ins
        del ins, ins_grouped
        gc.collect()
        
        return self
    
    def transform(self, X):
        df = X.copy()
        df = df.merge(self.ins_features, on='SK_ID_CURR', how='left')
        df.fillna(0, inplace=True)
        return df


class CreditCardBalanceFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer để xử lý Credit Card Balance data
    """
    def __init__(self, credit_card_path):
        self.credit_card_path = credit_card_path
        self.cc_features = None
        
    def fit(self, X, y=None):
        cc = pd.read_csv(self.credit_card_path)
        
        # Mark as fitted
        self.n_features_in_ = X.shape[1]
        
        # Features
        cc['LIMIT_USE'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
        cc['PAYMENT_DIV_MIN'] = cc['AMT_PAYMENT_CURRENT'] / (cc['AMT_INST_MIN_REGULARITY'] + 1)
        cc['LATE_PAYMENT'] = (cc['SK_DPD'] > 0).astype(int)
        cc['DRAWING_LIMIT_RATIO'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
        
        cc.replace([np.inf, -np.inf], np.nan, inplace=True)
        cc.fillna(0, inplace=True)
        
        # Aggregation
        num_cols = cc.select_dtypes(include=[np.number]).columns.tolist()
        num_cols.remove('SK_ID_PREV')
        num_cols.remove('SK_ID_CURR')
        
        num_agg = {col: ['max', 'mean', 'sum', 'var'] for col in num_cols}
        
        cc_agg = cc.groupby('SK_ID_CURR').agg(num_agg)
        cc_agg.columns = ['CR_' + '_'.join(col).upper() for col in cc_agg.columns]
        cc_agg.reset_index(inplace=True)
        cc_agg.fillna(0, inplace=True)
        
        self.cc_features = cc_agg
        
        # Cleanup cc
        del cc
        gc.collect()
        
        return self
    
    def transform(self, X):
        df = X.copy()
        df = df.merge(self.cc_features, on='SK_ID_CURR', how='left')
        df.fillna(0, inplace=True)
        return df


def create_feature_engineering_pipeline(data_path='home-credit-default-risk/'):
    """
    Tạo complete pipeline cho feature engineering
    
    Parameters:
    -----------
    data_path : str
        Đường dẫn đến thư mục chứa data files
    
    Returns:
    --------
    pipeline : Pipeline
        Sklearn pipeline hoàn chỉnh
    """
    import os
    
    # Ensure data_path ends with separator
    if not data_path.endswith(('/', '\\')):
        data_path = data_path + os.sep
    
    pipeline = Pipeline([
        ('null_outlier_fixer', NullOutlierFixer()),
        ('app_feature_engineer', ApplicationFeatureEngineer()),
        ('bureau_feature_engineer', BureauFeatureEngineer(
            bureau_path=data_path + 'bureau.csv',
            bureau_balance_path=data_path + 'bureau_balance.csv'
        )),
        ('prev_app_feature_engineer', PreviousApplicationFeatureEngineer(
            previous_app_path=data_path + 'previous_application.csv'
        )),
        ('pos_cash_feature_engineer', POSCashFeatureEngineer(
            pos_cash_path=data_path + 'POS_CASH_balance.csv'
        )),
        ('installments_feature_engineer', InstallmentsPaymentsFeatureEngineer(
            installments_path=data_path + 'installments_payments.csv'
        )),
        ('credit_card_feature_engineer', CreditCardBalanceFeatureEngineer(
            credit_card_path=data_path + 'credit_card_balance.csv'
        ))
    ])
    
    return pipeline
