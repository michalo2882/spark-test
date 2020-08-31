from pyspark.sql import SparkSession
from pyspark.sql.types import *

USER_SCHEMA = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), True),
    StructField("gender", StringType(), True)
])

ACCOUNT_SCHEMA = StructType([
    StructField("id", IntegerType(), False),
    StructField("user_id", IntegerType(), False),
    StructField("balance", IntegerType(), True)
])

TRANSACTION_SCHEMA = StructType([
    StructField("id", IntegerType(), False),
    StructField("timestamp", IntegerType(), True),
    StructField("value", IntegerType(), True),
    StructField("accounts_id", IntegerType(), False)
])


def run():
    spark = SparkSession \
        .builder \
        .appName("MyApp") \
        .getOrCreate()

    users = spark.read.csv("dataset/users.csv", schema=USER_SCHEMA, header=True)
    accounts = spark.read.csv("dataset/accounts.csv", schema=ACCOUNT_SCHEMA, header=True)
    transactions = spark.read.csv("dataset/transactions.csv", schema=TRANSACTION_SCHEMA, header=True)


if __name__ == '__main__':
    run()
