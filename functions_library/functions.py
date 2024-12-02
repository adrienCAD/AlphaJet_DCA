# imports
# Imports core Python libraries
import pandas as pd
import numpy as np

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool

from IPython.display import display
import hvplot.pandas

# Libraries for handling dates
from datetime import datetime, timedelta
import time
import pytz

# Libraries for handling system environment files and variables
import os
from dotenv import load_dotenv
from pathlib import Path

# Libraries for API calls
import requests  # HTTP library
import json      # JSON handling library

# Classification Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc #, plot_roc_curve
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Import models from scikit lear
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

# import technical analysis libraries
import finta as ft
from finta import TA
import talib




load_dotenv()
ccompare_api_key = os.getenv('CCOMPARE_API_KEY')

def get_crypto_OHLCV_cc(crypto='ETH', fiat='USD', limit =2000, toTS=-1,api_key = ccompare_api_key):
    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={crypto}&tsym={fiat}&limit={limit}&toTs={toTS}&api_key={api_key}"

    r = requests.get(url, headers={"User-Agent": "XY"})

    if r.status_code == 200:
        with open(f"resources/{crypto}{fiat}_ccompare.json", "wb") as file:
            file.write(r.content)
        print(f"{crypto}-{fiat} JSON data downloaded successfully.")
    else:
        print(f"Failed to download {crypto}-{fiat} data.")
        print(r)

    # # Load the JSON data from the file
    pair_json = Path(f"resources/{crypto}{fiat}_ccompare.json")
    list_pair_df = pd.read_json(pair_json)
    ohlcv_list = list_pair_df['Data']['Data']

    # Create a DataFrame
    pair_1H_df = pd.DataFrame(ohlcv_list)

    # Convert 'time' column to datetime format
    pair_1H_df['timestamp'] = pd.to_datetime(pair_1H_df['time'], unit='s')
    pair_1H_df.drop(columns=['time'], inplace=True)

    # Set 'time' as the index
    pair_1H_df.set_index('timestamp', inplace=True)
    pair_1H_df.index = pair_1H_df.index.tz_localize('UTC')

    # Rename columns to match OHLCV format
    pair_1H_df.rename(columns={'volumefrom': 'volume'}, inplace=True)

    # Reorder columns
    pair_1H_df = pair_1H_df[['open', 'high', 'low', 'close', 'volume']]

    return pair_1H_df

# Function to download and merge crypto OHLCV data from CryptoCompare

def download_and_merge_crypto_OHLCV_cc(crypto='ETH', fiat='USD'):
    # initialize variables and list to store dataframes
    limit=2000

    # Convert the current time to UTC
    now_utc = datetime.now(pytz.timezone('UTC'))

    # Check if the CSV file exists
    csv_path = f'Resources/{crypto}{fiat}_1H_ccompare.csv'

    if Path(csv_path).is_file():

        # Load the CSV file into a DataFrame
        result_df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')

        # Convert the latest timestamp to UTC
        latest_timestamp = result_df.index.max().tz_convert('UTC')
        first_timestamp = result_df.index.min().tz_convert('UTC')

        # Calculate the number of hours between now (in UTC) and the latest entry in the CSV file (in UTC)
        hours_since_latest = (now_utc - latest_timestamp).total_seconds() / 3600
        total_batches = int(hours_since_latest / limit) + 1
        hours_to_download = int(hours_since_latest)

        # Print information about the CSV file
        print(f"Detecting OHLCV data that was previously downloaded:")
        print(f" > latest timestamp available (UTC): {latest_timestamp}")
        print(f" > first timestamp available (UTC): {first_timestamp}")
        print("")

    else:
        # Initialize an empty DataFrame if the CSV file does not exist
        result_df = pd.DataFrame()

        # Calculate the number of batches needed from January 2017 until now
        start_date = datetime(2017, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        hours_since_start = (now_utc - start_date).total_seconds() / 3600
        hours_since_start = (now_utc - start_date).total_seconds() / 3600
        total_batches = int(hours_since_start / limit) + 1
        hours_to_download = int(hours_since_start)

    if hours_to_download == 0:
        print(f"Already up to date! No new {crypto}-{fiat} data to download.")

    else:
        print(f"Downloading {total_batches} batches of {limit} hours each from CryptoCompare.")
        print(f"for completing the {hours_to_download} missing hours of {crypto}-{fiat}")

        # We will start downloading from the latest timestamp available in the API (-1)
        batch_timestamp = -1

        for batch in range(total_batches):
            # Download data for the current batch
            df_batch = get_crypto_OHLCV_cc(crypto=crypto, fiat=fiat, limit=limit, toTS=batch_timestamp, api_key=ccompare_api_key)

            # Append the new batch to the DataFrame
            result_df = pd.concat([result_df, df_batch])

            # Check if the batch was downloaded successfully
            if not df_batch.empty:
                # Update the timestamp for the next batch
                earliest_timestamp = df_batch.index.min()
                latest_timestamp = df_batch.index.max()

                # Convert the datetime object to Unix time
                earliest_timestamp_unix = int(time.mktime(earliest_timestamp.timetuple()))
                batch_timestamp = earliest_timestamp_unix
                print(f"Batch {batch + 1}/{total_batches} downloaded successfully.")

            else:
                print(f"Failed to download batch {batch + 1}.")

    print(f"Latest {crypto}-{fiat} OHLCV dataset updated.")

    # Remove rows with missing values
    result_df.dropna(inplace=True)

    # Sort by timestamp and remove duplicates
    result_df = result_df.sort_index().drop_duplicates()

    # Checking if there is any misisng time steps
    ### Create a complete datetime index with 1-hour frequency
    complete_index = pd.date_range(start=result_df.index.min(), end=result_df.index.max(), freq='H')

    ### Find the difference between the complete index and the existing index
    missing_timestamps = complete_index.difference(result_df.index)

    if missing_timestamps.empty:
        print("No missing timestamps found.")

        # Save the resulting dataframe as both .csv and .json
        # When saving to CSV, reset the index
        result_df.reset_index().to_csv(csv_path, header=True, index=False)
        result_df.to_json(f'Resources/{crypto}{fiat}_1H_ccompare.json', orient='records', date_format='iso')

        print(f"All the following {crypto}-{fiat} OHLCV info has been saved to disk and is available now =)")
        DFinfo(result_df)

    else:
        print(f"{len(missing_timestamps)} missing timestamps found. Aborting - Please try again.")

    return result_df


def get_crypto_fear_and_greed():
    # Download Fear & Greed JSON data
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    response = requests.get(url)

    if response.status_code == 200:
        with open("resources/fear_and_greed_index.json", "wb") as file:
            file.write(response.content)
        print("Fear & Greed JSON data downloaded successfully.")
    else:
        print("Failed to download FNG data.")

    # Read the JSON data from the file
    fng_json = Path("resources/fear_and_greed_index.json")
    with open('resources/fear_and_greed_index.json') as file:
        data = json.load(file)

    # Extract the "data" section from the JSON
    data = data['data']

    # Convert the data to a DataFrame
    fng_df = pd.DataFrame(data)

    # Convert the "timestamp" column to datetime format
    fng_df['timestamp'] = pd.to_datetime(fng_df['timestamp'], unit='s')

    # Set the "timestamp" column as the index
    fng_df.set_index('timestamp', inplace=True)

    # Clean the dataframe by dropping the "time_until_update" column
    fng_df.drop("time_until_update",axis=1,inplace=True)

    # Resample the DataFrame to hourly frequency and forward fill missing values
    fng_df = fng_df.resample('1H').ffill()

    # rename column 'value' to 'fng'
    fng_df = fng_df.rename(columns={'value':'fng'})
    fng_df = fng_df.rename(columns={'value_classification':'fng_class'})

    # make 'fng' column an integer
    fng_df['fng'] = fng_df['fng'].astype(int)

    # Convert fng_df to time zone aware with UTC
    fng_df.index = fng_df.index.tz_localize('UTC')

    DFinfo(fng_df)

    return fng_df

def get_CBBI_index():
    # Download CBBI JSON data
    url_CBBI = "https://colintalkscrypto.com/cbbi/data/latest.json"
    r = requests.get(url_CBBI, headers={"User-Agent": "XY"})

    if r.status_code == 200:
        with open("resources/CBBI_index.json", "wb") as file:
            file.write(r.content)
        print("CBBI JSON data downloaded successfully.")
    else:
        print("Failed to download CBBI data.")
        print(r)

    # Load the JSON data from the file
    CBBI_json = Path("resources/CBBI_index.json")
    CBBI_df = pd.read_json(CBBI_json)

    CBBI_df = CBBI_df.reset_index(names='timestamp')

    # Convert the 'timestamp' column to a DatetimeIndex
    CBBI_df['timestamp'] = pd.to_datetime(CBBI_df['timestamp'], unit='s')
    CBBI_df.set_index('timestamp', inplace=True)

    # Resample the DataFrame to hourly frequency and forward fill missing values
    CBBI_df.resample('1H').ffill()

    # Cut off for 'Confidence'
    cutoffs = [-0.1, 0.25,0.4,0.50, 0.6,0.7,1.1]
    labels = ['X-Low','Low','Neutral', 'High', 'X-High', 'XX-High']

    CBBI_df['CBBI_class'] = pd.cut(CBBI_df['Confidence'], bins=cutoffs, labels=labels)

    # Convert fng_df and CBBI_df to time zone aware with UTC
    CBBI_df.index = CBBI_df.index.tz_localize('UTC')

    # Print 5 rows of resulting DataFrame
    DFinfo(CBBI_df)

    return (CBBI_df)


def DFinfo(df,n=3):
    display (df.head(n))
    display (df.tail(n))
    # display (df.info())
    return

# Detecting columns with NaN
def detect_Nan(df,threshold) :
    columns_with_nan = df.columns[df.isna().mean() > threshold]
    return(columns_with_nan)

def feature_engineering(merged_df):
    ohlcv_df = merged_df

    # List of time periods to use for Moving Averages calculation
    timeperiods = [30,60,100,200,300,500,750,1000,1500]

    df_eth_eng = ohlcv_df.copy()

    # Calculate SMAs and add them to the DataFrame
    for t in timeperiods:
        #tsma = TA.SMA(df_eth_eng t).shift(1)
        sma = TA.SMA(df_eth_eng,t)
        ema = TA.EMA(df_eth_eng,t)
        atr = TA.ATR(df_eth_eng,t)  #Average True Range
        adx = TA.ADX(df_eth_eng,t)
        rsi = TA.RSI(df_eth_eng,t)
        hma = TA.HMA(df_eth_eng,t)
        vama = TA.VAMA(df_eth_eng,t)

        # calculate the Force Index
        force_index = pd.Series(df_eth_eng['close'].diff(1) * df_eth_eng['volume'], index=df_eth_eng.index)
        force_ema = force_index.ewm(span=t, min_periods=0, adjust=True, ignore_na=False).mean()

        #df['force_index'] = force_index
        #df[f'force_index_ema_{t}'] = force_ema # add the Force Index and its EMA to the DataFrame
        #df[f'TSMA_{t}'] = tsma
        df_eth_eng[f'sma_{t}'] = sma
        df_eth_eng[f'ema_{t}'] = ema
        df_eth_eng[f'hma_{t}'] = hma
        df_eth_eng[f'vama_{t}'] = vama
        df_eth_eng[f'atr_{t}'] = atr
        df_eth_eng[f'adx_{t}'] = adx
        df_eth_eng[f'rsi_{t}'] = rsi


    # Calculate the Parabolic SAR
    #sar = TA.PSAR(df)

    # Add the SAR values and trend direction to the DataFrame
    #df['sar'] = sar['psar']
    #df['psarbear'] = sar['psarbear']
    #df['psarbull'] = sar['psarbull']

    df_eth_eng['uo'] = TA.UO(df_eth_eng)

    # Adding Awesome Indicator (AO)
    df_eth_eng['ao'] = TA.AO(df_eth_eng)
    df_eth_eng['obv'] =TA.OBV(df_eth_eng)

    # Adding Chaikin Indicator
    df_eth_eng['chaikin'] = TA.CHAIKIN(df_eth_eng)

    # Adding Bollinger Bands
    df_eth_eng[['bb_upper','bb_med','bb_lower']] =TA.BBANDS(df_eth_eng)

    # Calculate the Keltner Channel with TALIB
    #df[['KC_UPPER','KC_MED','KC_LOWER']] = TA.KC(df)

    # calculate Commodity Channel Index (CCI)
    df_eth_eng['cci'] = TA.CCI(df_eth_eng)

    # assuming you have OHLCV data in a pandas dataframe called "df"
    #volume_momentum = talib.MOM(df['volume'])

    # calculate the Ichimoku Kinko Hyo indicator
    # Calculate the conversion line
    nine_period_high = df_eth_eng['high'].rolling(window=9).max()
    nine_period_low = df_eth_eng['low'].rolling(window=9).min()
    df_eth_eng['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    # Calculate the base line
    periods = 26
    twenty_six_period_high = df_eth_eng['high'].rolling(window=periods).max()
    twenty_six_period_low = df_eth_eng['low'].rolling(window=periods).min()
    df_eth_eng['kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2

    # Calculate the leading span A
    df_eth_eng['senkou_span_a'] = ((df_eth_eng['tenkan_sen'] + df_eth_eng['kijun_sen']) / 2).shift(periods=periods)

    # Calculate the leading span B
    periods2 = 52
    fifty_two_period_high = df_eth_eng['high'].rolling(window=periods2).max()
    fifty_two_period_low = df_eth_eng['low'].rolling(window=periods2).min()
    df_eth_eng['senkou_span_b'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(periods=periods)

    # Calculate the lagging span
    #df_eth_eng['chikou_span'] = df_eth_eng['close'].shift(periods=-periods)

    # Drop columns with more than 10% NaN values
    columns_with_nan = detect_Nan(df_eth_eng,0.1)
    df_eth_eng_dropped = df_eth_eng.drop(columns=columns_with_nan, axis=1)

    # Updating the df_eth_eng DataFrame
    df_eth_eng = df_eth_eng_dropped.copy()

    # Calculate weekly % change columns and add them to df_eth_eng
    # 2 weeks to 9 months

    days = [14, 21, 28, 42, 56, 63, 84, 112, 140, 168, 196, 224]

    for day in days:
        column_name = f'{day}d_pct_change'

        # Calculate the percentage change for the specified day
        df_eth_eng[column_name] = df_eth_eng['close'].pct_change(periods=day).mul(100)

    df_eth_eng = df_eth_eng.dropna()

    # Convert the first character of column name to lowercase
    df_eth_eng.columns = [col[0].lower() + col[1:] for col in df_eth_eng.columns]

    return df_eth_eng

def pre_processing(df_eth_eng):
    df_feats = df_eth_eng.copy()
    df_feats = df_feats[['volume', 'fng', 'confidence','piCycle', 'sma_30','sma_60','sma_100', 'uo', 'ao', 'chaikin']]
    df_feats['fng_class'] = df_eth_eng['fng_class']
    df_feats['2w_returns'] = df_eth_eng['14d_pct_change']
    df_feats['1m_returns'] = df_eth_eng['28d_pct_change']
    df_feats['2m_returns'] = df_eth_eng['56d_pct_change']
    df_feats['3m_returns'] = df_eth_eng['84d_pct_change']
    df_feats['4m_returns'] = df_eth_eng['112d_pct_change']
    df_feats['5m_returns'] = df_eth_eng['140d_pct_change']
    df_feats['6m_returns'] = df_eth_eng['168d_pct_change']
    df_feats['7m_returns'] = df_eth_eng['196d_pct_change']
    # replace spaces with underscores in the categorical column
    df_feats["fng_class"] = df_feats["fng_class"].str.replace("Extreme ", "Hi_")
    # Get dummy variables for the categorical column
    dummy_df = pd.get_dummies(df_feats['fng_class'], prefix='fng')

    # Concatenate the original DataFrame with the dummy-encoded DataFrame
    df_eth_encoded = pd.concat([df_feats, dummy_df], axis=1)

    # Drop the original categorical column (optional)
    df_eth_encoded.drop('fng_class', axis=1, inplace=True)

    df_eth_encoded.dropna(inplace=True)
    return df_eth_encoded


def model_selection(X,Y):
    seed = 7
    models = []
    #models.append(('LogisticRegression', LogisticRegression(random_state=seed)))
    models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))

    models.append(('RandomForestClassifier', RandomForestClassifier()))
    models.append(('ExtraTreesClassifier',ExtraTreesClassifier(random_state=seed)))
    models.append(('AdaBoostClassifier',AdaBoostClassifier(
                      DecisionTreeClassifier(random_state=seed),random_state=seed,learning_rate=0.1))
                 )
    models.append(('SVM',svm.SVC(random_state=seed)))
    models.append(('GradientBoostingClassifier',GradientBoostingClassifier(random_state=seed)))
    models.append(('XGBoost', xgb.XGBClassifier(random_state=seed)))
    #models.append(('CatBoost', CatBoostClassifier(iterations=100,l2_leaf_reg=2,random_state=seed)))
    #models.append(('MLPClassifier',MLPClassifier(random_state=seed)))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'f1'

    for name, model in models:
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, X,Y, cv=kfold, scoring=scoring,verbose=False)
        results.append([name, round(cv_results.mean(),2), round(cv_results.std(),3)])
        names.append(name)

    results_df = pd.DataFrame(results, columns=['Model Name', 'F1_Mean', 'F1_Standard Deviation'])
    return results_df

def Backtesting (df, X_test, X_test_scaled, classifier, trading_fee):
    # Create a new empty predictions DataFrame using code provided below.
    classifier_df = pd.DataFrame(index=X_test.index)
    classifier_df["predicted_signal"] = classifier.predict(X_test_scaled)
    classifier_df["actual_returns"] = df["close"].pct_change()

    # Calculate the algo-trading returns without fee
    classifier_df["trading_algorithm_returns"] = classifier_df["actual_returns"] * classifier_df["predicted_signal"]

    # Drop all NaN values from the DataFrame
    classifier_df = classifier_df.dropna()

    # Create a mask to define the rows where to apply the transaction fee
    trading_fee_mask = classifier_df["predicted_signal"].diff() != 0

    # Apply the fee to the previous row's actual return with shift(1)
    classifier_df.loc[trading_fee_mask, "trading_algorithm_returns"] -= trading_fee * classifier_df["actual_returns"].shift(1)

    # Sort the DataFrame by index
    classifier_df = classifier_df.sort_index()
    return classifier_df


def ROC(classifier,X_train,X_test,y_train,y_test) :

    # set the increment step
    inc = .2

    # get the predicted probabilities of the positive class
    y_score_train = classifier.predict_proba(X_train)[:,1]
    y_score_test = classifier.predict_proba(X_test)[:,1]

    # calculate y_train and y_test
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    # calculate the fpr, tpr and thresholds for each increment
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_score_test)
    tpr_test_smooth = []
    fpr_test_smooth = []

    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_score_train)
    tpr_train_smooth = []
    fpr_train_smooth = []


    # interpolate the ROC curve at each increment
    for i in np.arange(0, 1, inc):
        tpr_test_smooth.append(np.interp(i, fpr_test, tpr_test))
        fpr_test_smooth.append(i)
        tpr_train_smooth.append(np.interp(i, fpr_train, tpr_train))
        fpr_train_smooth.append(i)

    # fixing the low and high end points of the ROC curve
    tpr_train_smooth[0] = 0
    tpr_test_smooth[0] = 0
    fpr_train_smooth[0] = 0
    fpr_test_smooth[0] = 0

    tpr_train_smooth[-1] = 1
    tpr_test_smooth[-1] = 1
    fpr_train_smooth[-1] = 1
    fpr_test_smooth[-1] = 1

    # calculate AUC
    roc_auc_test = auc(fpr_test_smooth, tpr_test_smooth)
    roc_auc_train = auc(fpr_train_smooth, tpr_train_smooth)

    # Set the plot size
    plt.subplots(figsize=(8,5))

    # plot the ROC curve with smooth interpolation
    plt.plot(fpr_test_smooth, tpr_test_smooth, lw=2, label='ROC curve for TEST (AUC = %0.2f)' % roc_auc_test)
    plt.plot(fpr_train_smooth, tpr_train_smooth, lw=2, label='ROC curve for TRAIN (AUC = %0.2f)' % roc_auc_train)

    # plot the random guessing line
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guess')

    # set plot title and labels
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    # show the plot
    plt.show()

    # Evaluate the model using a classification report
    training_report = classification_report(y_train, y_train_pred,zero_division=1)
    print("TRAINING classification report: \n",training_report)

    # Evaluate the model using a classification report
    testing_report = classification_report(y_test, y_test_pred,zero_division=1)
    print("\nTESTING classification report: \n",testing_report)

    # Calculate the accuracy, precision, F1 and recall of the model
    f1 = f1_score(y_test, y_test_pred, average='micro')
    accuracy= accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='micro',zero_division=1)
    recall = recall_score(y_test, y_test_pred, average='micro')

    # Print the results
    print(f'Test accuracy: {accuracy:.2f}')
    print(f'>> Test precision: {precision:.2f} <<')
    print(f'Test recall: {recall:.2f}')
    print(f'Test F1 score: {f1:.2f}')
    print(f'Test AUC score: {roc_auc_test:.2f}')


def plot_btcusd_confidence(df, confidence, sma1, sma2=None):

    # Assuming df is your DataFrame
    hv.extension('bokeh')

    # Scatter plot for 'close' and 'confidence'
    scatter_plot = hv.Scatter(
        data=df,
        kdims=['timestamp'],
        vdims=['close', confidence]
    ).opts(
        cmap='RdYlGn_r',
        color=confidence,
        size=5,
        clim=(df[confidence].min(), df[confidence].max()),
        title=f'Crypto Close Price with Confidence Coloring with {sma1}' ,
        xlabel='Time',
        ylabel='Crypto Close Price',
        colorbar=True,
        width=800,
        height=500)

    # Curve for 'close' in fine grey
    close_curve = hv.Curve(df[['close']], kdims=['timestamp'], vdims=['close']).opts(color='grey', line_width=1)

    # Curve for 'sma1' in fine blue
    sma_curve = hv.Curve(df[[sma1]], kdims=['timestamp'], vdims=[sma1]).opts(color='blue', line_width=2)

    # Check if sma2 is provided and create a curve for it
    if sma2:
        # Curve for 'sma2' in fine green
        sma_curve2 = hv.Curve(df[[sma2]], kdims=['timestamp'], vdims=[sma2]).opts(color='green', line_width=2)
        # Overlay the Scatter plot with all three Curves
        overlay_plot = scatter_plot * close_curve * sma_curve * sma_curve2
        overlay_plot.opts(
            opts.Scatter(tools=['hover'], size=5),
            # opts.Curve(tools=['hover']),
            # opts.Curve(tools=['hover']),
            # opts.Curve(tools=['hover'])
            )


    else:
        # Overlay the Scatter plot with only two Curves (without sma_curve2)
        overlay_plot = scatter_plot * close_curve * sma_curve
        overlay_plot.opts(
        opts.Scatter(tools=['hover'], size=5),
        # opts.Curve(tools=['hover']),
        # opts.Curve(tools=['hover'])
        )

    return overlay_plot
