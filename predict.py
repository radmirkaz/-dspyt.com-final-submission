import argparse
import logging.config
import pandas as pd
import numpy as np
from raif_hack.features import prepare_categorical
from traceback import format_exc

from raif_hack.model import BenchmarkModel
from raif_hack.settings import LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES, \
    CATEGORICAL_STE_FEATURES

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для предсказания модели
     
     Примеры:
        1) с poetry - poetry run python3 predict.py --test_data /path/to/test/data --model_path /path/to/model --output /path/to/output/file.csv.gzip
        2) без poetry - python3 predict.py --test_data /path/to/test/data --model_path /path/to/model --output /path/to/output/file.csv.gzip
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # parser.add_argument("--test_data", "-d", type=str, dest="d", required=True, help="Путь до отложенной выборки")
    # parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True,
    #                     help="Пусть до сериализованной ML модели")
    # parser.add_argument("--output", "-o", type=str, dest="o", required=True, help="Путь до выходного файла")

    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START predict.py')
        # args = vars(parse_args())
        logger.info('Load test df')
        test_df = pd.read_csv('C:/Users/Radmir/Desktop/raifhack/data/test.csv')
        logger.info(f'Input shape: {test_df.shape}')
        test_df = prepare_categorical(test_df)
        test_df['per_square_meter_price'] = np.ones(test_df.shape[0])

        logger.info('Load model')

        for fold in range(5):
            model = BenchmarkModel.load(f'C:/Users/Radmir/Desktop/raifhack/model/model{fold}_lgb_ohe.pkl')
            logger.info('Predict')
            test_df['per_square_meter_price'] += model.predict(test_df[NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]) / 5
        logger.info('Save results')
        test_df[['id','per_square_meter_price']].to_csv('C:/Users/Radmir/Desktop/raifhack/submission_lgb_ohe.csv', index=False)

        logger.info('Load model')

        for fold in range(5):
            model = BenchmarkModel.load(f'C:/Users/Radmir/Desktop/raifhack/model/model{fold}_xgb_ohe.pkl')
            logger.info('Predict')
            test_df['per_square_meter_price'] += model.predict(
                test_df[NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]) / 5
        logger.info('Save results')
        test_df[['id', 'per_square_meter_price']].to_csv('C:/Users/Radmir/Desktop/raifhack/submission_xgb_ohe.csv',
                                                         index=False)

        logger.info('Load model')

        for fold in range(5):
            model = BenchmarkModel.load(f'C:/Users/Radmir/Desktop/raifhack/model/model{fold}_cat_ohe.pkl')
            logger.info('Predict')
            test_df['per_square_meter_price'] += model.predict(
                test_df[NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]) / 5
        logger.info('Save results')
        test_df[['id', 'per_square_meter_price']].to_csv('C:/Users/Radmir/Desktop/raifhack/submission_cat_ohe.csv',
                                                         index=False)
    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise (e)

    logger.info('END predict.py')