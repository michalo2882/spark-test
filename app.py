from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col, row_number
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


def group_by_top3_accounts(accounts):
    window = Window.partitionBy(accounts.user_id).orderBy(accounts.balance.desc())
    df = accounts.select("*", rank().over(window).alias('rank'))
    df = df.filter(df['rank'] <= 3)
    return df.groupBy("user_id").pivot("rank").max("balance").na.fill(-1) \
        .withColumnRenamed("1", "balance_account_0") \
        .withColumnRenamed("2", "balance_account_1") \
        .withColumnRenamed("3", "balance_account_2")


def create_name_labels(users):
    return users.select("name").distinct().na.drop().orderBy("name").withColumn("index", row_number().over(
        Window.orderBy("name")) - 1)


def create_gender_labels(users):
    return users.select("gender").distinct().na.drop().orderBy("gender").withColumn("index", row_number().over(
        Window.orderBy("gender")) - 1)


def join_accounts_and_users(accounts, users, name_labels, gender_labels):
    return users.join(accounts, users.id == accounts.user_id).drop("user_id") \
        .join(name_labels, users.name == name_labels.name, 'left').drop("name") \
        .withColumnRenamed("index", "name") \
        .join(gender_labels, users.gender == gender_labels.gender, 'left').drop("gender") \
        .withColumnRenamed("index", "gender") \
        .select("id", "name", "gender", "balance_account_0", "balance_account_1", "balance_account_2")


def join_accounts_and_transactions(transactions, accounts):
    window = Window.partitionBy(accounts.user_id).orderBy(accounts.id)
    accounts = accounts.select("id", "user_id", (rank().over(window) - 1).alias('account_number'))
    return transactions.alias('t') \
        .join(accounts.alias('user'), accounts.id == transactions.accounts_id) \
        .orderBy("user_id", "timestamp") \
        .select("t.id", "user_id", "timestamp", "value", "account_number")


def run():
    spark = SparkSession \
        .builder \
        .appName("MyApp") \
        .getOrCreate()

    users = spark.read.csv("dataset/users.csv", schema=USER_SCHEMA, header=True)
    accounts = spark.read.csv("dataset/accounts.csv", schema=ACCOUNT_SCHEMA, header=True)
    transactions = spark.read.csv("dataset/transactions.csv", schema=TRANSACTION_SCHEMA, header=True)


    accounts = accounts.na.drop()
    transactions = transactions.na.drop()

    name_labels, gender_labels = create_name_labels(users), create_gender_labels(users)
    filtered_accounts = group_by_top3_accounts(accounts)

    user_accounts = join_accounts_and_users(filtered_accounts, users, name_labels, gender_labels)

    transaction_and_accounts = join_accounts_and_transactions(transactions, accounts)

    name_labels.write.parquet("name-labels.parquet", mode='overwrite')
    gender_labels.write.parquet("gender-labels.parquet", mode='overwrite')
    user_accounts.write.parquet("user-accounts.parquet", mode='overwrite')
    transaction_and_accounts.write.parquet("transactions.parquet", mode='overwrite')


if __name__ == '__main__':
    run()
