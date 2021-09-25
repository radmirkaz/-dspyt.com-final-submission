import argparse
import logging.config

import catboost
import pandas as pd
from traceback import format_exc

import xgboost

from raif_hack.model import BenchmarkModel
from raif_hack.settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET
from raif_hack.utils import PriceTypeEnum
from raif_hack.metrics import metrics_stat
from raif_hack.features import prepare_categorical
from sklearn.model_selection import KFold
import lightgbm
from raif_hack.settings import xgb_params

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

nnn = CATEGORICAL_OHE_FEATURES + NUM_FEATURES

def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для обучения модели
     
     Примеры:
        1) с poetry - poetry run python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        2) без poetry - python3 train.py --train_data /path/to/train/data --model_path /path/to/model
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    #
    # parser.add_argument("--train_data", "-d", type=str, dest="d", required=True, help="Путь до обучающего датасета")
    # parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True, help="Куда сохранить обученную ML модель")

    return parser.parse_args()


if __name__ == "__main__":
     # making agg features
    try:
        # logger.info('START train.py')
        # args = vars(parse_args())
        # logger.info('Load train df')
        train_df = pd.read_csv('C:/Users/Radmir/Desktop/raifhack/data/train.csv')
        logger.info(f'Input shape: {train_df.shape}')
        train_df = prepare_categorical(train_df)

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (trn_ind, val_ind) in enumerate(kfold.split(train_df)):
            X_offer = train_df.iloc[trn_ind][train_df.price_type == PriceTypeEnum.OFFER_PRICE][
                NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]
            y_offer = train_df.iloc[trn_ind][train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
            X_manual = train_df.iloc[val_ind][train_df.price_type == PriceTypeEnum.MANUAL_PRICE][
                NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]
            y_manual = train_df.iloc[val_ind][train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
            model = BenchmarkModel(numerical_features=NUM_FEATURES,
                                   ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                   # ste_categorical_features=CATEGORICAL_STE_FEATURES,
                                   model_params=MODEL_PARAMS, model=lightgbm.LGBMRegressor)

            print('fitting')
            model.fit(X_offer, y_offer, X_manual, y_manual)
            print('saving')
            model.save(f'C:/Users/Radmir/Desktop/raifhack/model/model{fold}_lgb_ohe.pkl')

            predictions_offer = model.predict(X_offer)
            metrics = metrics_stat(y_offer.values, predictions_offer / (
                        1 + model.corr_coef))  # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
            print(metrics)
            predictions_manual = model.predict(X_manual)
            metrics = metrics_stat(y_manual.values, predictions_manual)
            print(metrics)

            model = BenchmarkModel(numerical_features=NUM_FEATURES,
                                   ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                   # ste_categorical_features=CATEGORICAL_STE_FEATURES,
                                   model_params=xgb_params, model=xgboost.XGBRegressor)

            print('fitting')
            model.fit(X_offer, y_offer, X_manual, y_manual)
            print('saving')
            model.save(f'C:/Users/Radmir/Desktop/raifhack/model/model{fold}_xgb_ohe.pkl')

            predictions_offer = model.predict(X_offer)
            metrics = metrics_stat(y_offer.values, predictions_offer / (
                    1 + model.corr_coef))  # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
            print(metrics)
            predictions_manual = model.predict(X_manual)
            metrics = metrics_stat(y_manual.values, predictions_manual)
            print(metrics)

            model = BenchmarkModel(numerical_features=NUM_FEATURES,
                                   ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                   # ste_categorical_features=CATEGORICAL_STE_FEATURES,
                                   model=catboost.CatBoostRegressor)

            print('fitting')
            model.fit(X_offer, y_offer, X_manual, y_manual)
            print('saving')
            model.save(f'C:/Users/Radmir/Desktop/raifhack/model/model{fold}_cat_ohe.pkl')

            predictions_offer = model.predict(X_offer)
            metrics = metrics_stat(y_offer.values, predictions_offer / (
                    1 + model.corr_coef))  # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
            print(metrics)
            predictions_manual = model.predict(X_manual)
            metrics = metrics_stat(y_manual.values, predictions_manual)
            print(metrics)

    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')