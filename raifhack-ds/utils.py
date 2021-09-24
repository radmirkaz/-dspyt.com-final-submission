from enum  import IntEnum

UNKNOWN_VALUE = 'missing'

class PriceTypeEnum(IntEnum):

    OFFER_PRICE = 0 # цена из объявления
    MANUAL_PRICE = 1 # цена, полученная путем ручной оценки