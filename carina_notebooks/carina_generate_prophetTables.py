# Databricks notebook source
# MAGIC %pip install -U prophet
# MAGIC %pip install -U data_utils
# MAGIC %pip install mlflow

# COMMAND ----------

from snowflake_client import SnowflakeClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time
from databricks import koalas as ks
from pyspark.sql.types import *
from pyspark.sql.functions import current_date
import matplotlib.pyplot as plt
from pyspark.sql.functions import pandas_udf, PandasUDFType
from prophet import Prophet

# COMMAND ----------

client = SnowflakeClient('refined_spins')
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', None)

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
ip_total_us = client.query("SELECT * FROM IMMUTA.REFINED_SPINS.ABT_SPINS  " ) #and UPC = '00-17800-12596'
# where UPC = '00-17800-12596'
# where MARKET_REGION = 'MARKET' and UPC = '00-17800-12596' 
ip_pandas_total_us = ks.DataFrame(ip_total_us)
ip_pandas = ip_pandas_total_us

ip_pandas = ip_pandas[ip_pandas['BRAND'] != 'PRIVATE LABEL']
ip_pandas['SPPD'].fillna(0, inplace=True)
ip_pandas = ip_pandas[ip_pandas['SPPD'] > 0]

# COMMAND ----------

ip_pandas.shape

# COMMAND ----------

prod = client.table("ABT_SPINS_PRODUCTS")
prodPandas = prod.toPandas()

del prodPandas['MAIN_CATEGORY']
del prodPandas['SUBCATEGORY']
del prodPandas['BRAND']
del prodPandas['DESCRIPTION']

# COMMAND ----------

result_schema = StructType([
  StructField('ds',TimestampType()),
  StructField('UPC',StringType()),
  StructField('MARKET_REGION',StringType()),
  StructField('MARKET_REGION_NAME',StringType()),
  StructField('CHANNEL',StringType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  StructField('yhat_upper',FloatType()),
  StructField('yhat_lower',FloatType())
])    

@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def forecast_store_item( history_pd: pd.DataFrame ) -> pd.DataFrame:

  # TRAIN MODEL AS BEFORE
  # --------------------------------------
  # remove missing values 
  history_pd = history_pd.dropna()

  # configure the model
  model = Prophet(weekly_seasonality=True,seasonality_mode='additive')
  model.add_country_holidays(country_name='US')
  # train the model
  model.fit( history_pd )
  # --------------------------------------

  # BUILD FORECAST AS BEFORE
  # --------------------------------------
  # make predictions
  future_pd = model.make_future_dataframe(
    periods=60, 
    freq='W', 
    include_history=True
    )

  forecast_pd = model.predict( future_pd )  
  # --------------------------------------

  # ASSEMBLE EXPECTED RESULT SET
  # --------------------------------------
  # get relevant fields from forecast
  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')

  # get relevant fields from history
  h_pd = history_pd[['ds','MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC','y']].set_index('ds')

  # join history and forecast
  results_pd = f_pd.join( h_pd, how='left' )
  results_pd.reset_index(level=0, inplace=True)

  # get store & item from incoming data set
  results_pd['UPC'] = history_pd['UPC'].iloc[0]
  results_pd['MARKET_REGION'] = history_pd['MARKET_REGION'].iloc[1]
  results_pd['MARKET_REGION_NAME'] = history_pd['MARKET_REGION_NAME'].iloc[2]
  results_pd['CHANNEL'] = history_pd['CHANNEL'].iloc[3]
  # --------------------------------------

  # return expected dataset
  return results_pd[ ['ds', 'MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC', 'y', 'yhat', 'yhat_upper', 'yhat_lower'] ]    

class AlphaModel():
    def __init__(self, TARGET, LEVEL):
        # Example how to work with alpha's state:
        # self.observed_states = []
        # prod prep

        self.TARGET = TARGET
        self.LEVEL = LEVEL
        self.params ={'iterations': 2000 }
        
        prod = prodPandas.filter(
            ['UPC', 'PRODUCT_TYPE', 'POSITIONING_GROUP', 'LABELED_ORGANIC', 'SUBCATEGORY','BRAND', 
             'COMPANY',  'UNIT_OF_MEASURE',  'PACKAGING_TYPE_PRIMARY', 'FORM', 'LABELED_NON_GMO', 
             'STORAGE', 'PACK_COUNT'])
        self.prod = prod

        self.ip=ip_pandas    
        
        if self.LEVEL == "TOTAL US":
          self.ip = self.ip[self.ip['MARKET_REGION'] == 'TOTAL US']
        elif self.LEVEL == "MARKET":
          self.ip = self.ip[self.ip['MARKET_REGION'] == 'MARKET']
        elif self.LEVEL == "REGION":
          self.ip = self.ip[self.ip['MARKET_REGION'] == 'REGION']    
 
        
        self.ip =  self.ip[ self.ip['SPPD'] > 0]
        self.ip['DISCOUNT_PERC'] =  self.ip['DISCOUNT_PERC'].replace(np.nan, 0)

        # MERGING PRODUCT ATTRIBUTES
        self.ip =  self.ip.merge(ks.DataFrame(prodPandas), how="left", on=['UPC'])
        self.ip['UPC'].fillna(0, inplace=True)     
        self.ip['C19_NYCASES'] = self.ip['C19_NYCASES'].fillna(value=0)
        self.ip['C19_NYCASES'] = self.ip['C19_NYCASES'].astype(float)
        self.ip['C19_DEATHS'] = self.ip['C19_DEATHS'].fillna(value=0)
        self.ip['C19_DEATHS'] = self.ip['C19_DEATHS'].astype(float) 
        self.ip['INITIAL_CLAIMS'] = self.ip['INITIAL_CLAIMS'].fillna(value=0)
        self.ip['INITIAL_CLAIMS'] = self.ip['INITIAL_CLAIMS'].astype(float)
        self.ip['TIMEPERIODENDDATE'] = ks.to_datetime(self.ip['TIMEPERIODENDDATE'])
        self.ip['WEEK']=self.ip['TIMEPERIODENDDATE'].dt.week
        self.ip['WEEK'] = self.ip['WEEK'].astype(int)
        self.ip['MONTH']=self.ip['TIMEPERIODENDDATE'].dt.month
        self.ip['MONTH'] = self.ip['MONTH'].astype(int)
        self.ip['YEAR']=self.ip['TIMEPERIODENDDATE'].dt.year
        self.ip['YEAR'] = self.ip['YEAR'].astype(int)
             
        
        
        week_count= self.ip.groupby(by=['MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC'],as_index=False)["WEEK"].count()
        week_count.rename(columns = {'WEEK':'WEEK_COUNT'}, inplace = True)
        
        self.ip=self.ip.merge(week_count,on=['MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC'],how='left')

        self.ip = self.ip[self.ip['WEEK_COUNT'] > 12]
                          
        self.dft = self.ip[['MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC','TIMEPERIODENDDATE','SPPD']]
        self.dft.rename(columns = {'TIMEPERIODENDDATE':'ds'}, inplace = True)
        self.dft.rename(columns = {'SPPD':'y'}, inplace = True)
        self.dft=self.dft.to_spark()
        



        results_prp = (
            self.dft
                .groupBy('MARKET_REGION','MARKET_REGION_NAME','CHANNEL', 'UPC')
                  .apply(forecast_store_item)
                .withColumn('training_date', current_date())
            )
        


        forecast=ks.DataFrame(results_prp)
        forecast=forecast.filter(items=['ds','yhat','MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC'])
        forecast.rename(columns={'ds':'TIMEPERIODENDDATE'},inplace=True)

        min_sppd=self.ip.filter(items=['MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC','TIMEPERIODENDDATE','SPPD'])
        min_sppd=min_sppd[min_sppd.TIMEPERIODENDDATE<='2021-12-31']
        min_sppd=min_sppd[min_sppd.TIMEPERIODENDDATE>='2021-01-01']
        min_sppd=min_sppd.groupby(['MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC'])['SPPD'].min().reset_index()
        min_sppd.rename(columns = {'SPPD':'min_sppd'}, inplace = True)
        min_sppd=min_sppd.set_index(['MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC'])

        max_sppd=self.ip.filter(items=['MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC','TIMEPERIODENDDATE','SPPD'])
        max_sppd=max_sppd[max_sppd.TIMEPERIODENDDATE<'2021-12-31']
        max_sppd=max_sppd[max_sppd.TIMEPERIODENDDATE>='2021-01-01']
        max_sppd=max_sppd.groupby(['MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC'])['SPPD'].max().reset_index()
        max_sppd.rename(columns = {'SPPD':'max_sppd'}, inplace = True)        
        max_sppd=max_sppd.set_index(['MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC'])      
        forecast=forecast.set_index(['MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC'])


      
        forecast = ks.merge(forecast,min_sppd, left_index=True, right_index=True,how='left')
        forecast = ks.merge(forecast,max_sppd, left_index=True, right_index=True,how='left')

        forecast=forecast.reset_index()
        forecast = forecast[['MARKET_REGION','MARKET_REGION_NAME','CHANNEL','UPC',  'TIMEPERIODENDDATE', 'yhat','min_sppd', 'max_sppd']]
        



        if self.LEVEL == "TOTAL US":
          client.save((forecast.to_spark()), 'carina_prophet_features_TOTALUS_deploy', 'OVERWRITE')
        elif self.LEVEL == "MARKET":
          client.save((forecast.to_spark()), 'carina_prophet_features_MARKET_deploy', 'OVERWRITE')
        elif self.LEVEL == "REGION":
          client.save((forecast.to_spark()), 'carina_prophet_features_REGION_deploy', 'OVERWRITE') 


     
      
      

model_market = AlphaModel('SPPD', 'REGION')
