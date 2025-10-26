#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import numpy as np
import requests
import pandas as pd
import preprocessing as pp
from datetime import datetime, timedelta

START_TIME = (datetime.now() - timedelta(hours=0, minutes=1))
API_BASE = 'https://api.bybit.com/v5/'
DEBUG_MODE = True

LABELS = [
    'open_time',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'quote_asset_volume',
]

category = 'spot'

def get_batch(symbol, interval='1', start_time=0, limit=1000):
    """Use a GET request to retrieve a batch of candlesticks. Process the JSON into a pandas
    dataframe and return it. If not successful, return an empty dataframe.
    """

    params = {
        'category': category,
        'symbol': symbol,
        'interval': interval,
        'start': start_time,
        'limit': limit
    }
    response_json = None
    try:
        # timeout should also be given as a parameter to the function
        response = requests.get(f'{API_BASE}market/kline', params, timeout=30)
        response_json = response.json()
        data = response_json['result']['list']
        data.reverse()

    except requests.exceptions.ConnectionError:
        print('Connection error, Cooling down for 5 mins...')
        time.sleep(5 * 60)
        return get_batch(symbol, interval, start_time, limit)
    
    except requests.exceptions.Timeout:
        print('Timeout, Cooling down for 5 min...')
        time.sleep(5 * 60)
        return get_batch(symbol, interval, start_time, limit)
    
    except ConnectionResetError:
        print('Connection reset by peer, Cooling down for 5 min...')
        time.sleep(5 * 60)
        return get_batch(symbol, interval, start_time, limit)

    except Exception as e:
        print(f'Unknown error: {e}, for response: {response_json}')
        raise e

    if response.status_code == 200:
        df = pd.DataFrame(data, columns=LABELS)
        df = df.drop(columns=['quote_asset_volume'])
        df['open_time'] = df['open_time'].astype(np.int64)
        df = df[df.open_time < START_TIME.timestamp() * 1000]
        return df

    print(f'Got erroneous response back: {response}, {response_json}')
    return pd.DataFrame([])

def append_stored_candles(base, quote, interval='1'):
    """Collect a list of candlestick batches with all candlesticks of a trading pair,
    concat into a dataframe and write it to parquet.
    """

    # load from disk or start fresh
    batches = [pd.DataFrame([], columns=LABELS)]
    try:
        parquet_path = f'compressed/{base}-{quote}.parquet'
        df_disk = pd.read_parquet(parquet_path)
        pp.debug("Data from disk:")
        pp.debug(df_disk)
        last_timestamp = int(df_disk['datetime'].max().timestamp() * 1000)
        new_file = False
        batches = [df_disk]
    except Exception:
        # last_timestamp = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
        last_timestamp = 1640995200000  # 2022-01-01
        new_file = True
    old_lines = len(batches[-1])

    # gather all candlesticks available, starting from the last timestamp loaded from disk or 0
    # stop if the timestamp that comes back from the api is the same as the last one
    previous_timestamp = None

    while previous_timestamp != last_timestamp:
        # stop if we reached data from today
        #if date.fromtimestamp(last_timestamp / 1000) >= date.today():
        #    break

        previous_timestamp = last_timestamp

        new_batch = get_batch(
            symbol=base+quote,
            interval=interval,
            start_time=last_timestamp+1
        )

        # requesting candles from the future returns empty
        # also stop in case response code was not 200
        if new_batch.empty and not new_file:
            break

        if new_batch.empty:
            last_timestamp = last_timestamp + 86400000
        else:
            last_timestamp = new_batch['open_time'].max()

        # sometimes no new trades took place yet on date.today();
        # in this case the batch is nothing new
        if previous_timestamp == last_timestamp:
            break

        if not new_batch.empty:
            new_batch['datetime'] = pd.to_datetime(new_batch['open_time'], unit='ms', utc=True)
            batches.append(new_batch)
            new_file = False
        last_datetime = datetime.fromtimestamp(last_timestamp / 1000)

        covering_spaces = 20 * ' '
        print(datetime.now().replace(microsecond=0), base, quote, interval + 'min', str(last_datetime)+covering_spaces, end='\r', flush=True)

    # write clean version of csv to parquet
    parquet_name = f'{base}-{quote}.parquet'
    full_path = f'compressed/{parquet_name}'
    df = pd.concat(batches, ignore_index=True)
    pp.debug("Before cleaning:")
    pp.debug(df)
    df = pp.clean(df)
    pp.write_raw_to_parquet(df, full_path)

    if len(batches) > 1:
        return len(df) - old_lines
    return 0


def main():
    """Main loop; loop over all currency pairs that exist on the exchange. Once done upload the
    compressed (Parquet) dataset to Kaggle.
    """
    global category
    if len(sys.argv) > 1:
        category = sys.argv[1]

    # pp.groom_all()
    # exit()

    # get all pairs currently available
    all_symbols = pd.DataFrame(requests.get(f'{API_BASE}market/instruments-info?category=' + category).json()['result']['list'])
    all_symbols = all_symbols[all_symbols['quoteCoin'].isin(['USDT', 'USDC'])]
    blacklist = ['EUR', 'GBP', 'AUD', 'BCHABC', 'BCHSV', 'DAI', 'PAX', 'WBTC', 'BUSD', 'TUSD', 'UST', 'USDC', 'USDSB', 'USDS', 'SUSD', 'USDP']
    for coin in blacklist:
        all_symbols = all_symbols[all_symbols['baseCoin'] != coin]
    all_pairs = [tuple(x) for x in all_symbols[['baseCoin', 'quoteCoin']].to_records(index=False)]
    
    # sort reverse alphabetical, to ensure USDT pairings are updated first
    all_pairs.sort(key=lambda x:x[1], reverse=True)
    filtered_pairs = []
    for pair in all_pairs:
        if pair[0][-2:] == '2L' or pair[0][-2:] == '3L' or pair[0][-2:] == '2S' or pair[0][-2:] == '3S':
            print("Skipping", pair[0])
        else:
            filtered_pairs.append(pair)

    # debug, only one pair
    if DEBUG_MODE:
        filtered_pairs = [('BTC', 'USDT')]

    # make sure data folders exist
    os.makedirs('compressed', exist_ok=True)

    # do a full update on all pairs
    n_count = len(filtered_pairs)
    for n, pair in enumerate(filtered_pairs, 1):
        try:
            base, quote = pair
            new_lines = append_stored_candles(base=base, quote=quote)
            if new_lines > 0:
                print(f'{datetime.now()} {n}/{n_count} Wrote {new_lines} new lines to file for {base}-{quote}')
            else:
                print(f'{datetime.now()} {n}/{n_count} Already up to date with {base}-{quote}')
        except Exception as e:
            print(f'Error processing pair {pair}: {repr(e)}')


if __name__ == '__main__':
    main()
