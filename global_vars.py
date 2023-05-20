import time
from datetime import datetime

TODAY_TIMESTAMP = int(time.time())
TODAY_DAY = datetime.today().strftime('%d')
TODAY_MONTH = datetime.today().strftime('%m')
TODAY_YEAR = datetime.today().strftime('%Y')

MONTHS = {
    'январь': '01',
    'февраль': '02',
    'март': '03',
    'апрель': '04',
    'май': '05',
    'июнь': '06',
    'июль': '07',
    'август': '08',
    'сентябрь': '09',
    'октябрь': '10',
    'ноябрь': '11',
    'декабрь': '12',
    'янв': '01',
    'фев': '02',
    'мар': '03',
    'апр': '04',
    'июн': '06',
    'июл': '07',
    'авг': '08',
    'сен': '09',
    'окт': '10',
    'ноя': '11',
    'дек': '12',
}
