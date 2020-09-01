import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
os.environ['SPARK_HOME'] = '/home/michalo/spark-demo/spark-3.0.0-bin-hadoop2.7'
os.environ['HADOOP_HOME'] = os.environ['SPARK_HOME']

import findspark
findspark.init()

import app
app.run()
