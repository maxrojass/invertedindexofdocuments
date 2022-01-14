%%time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
from nltk.stem.snowball import SnowballStemmer
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.appName('MondraAssignment').getOrCreate()
path = "dataset"
inputData = sc.wholeTextFiles(path)
dd = inputData.toDF()
dd= dd.selectExpr("_1 as Filepath", "_2 as Document")
dd = dd.withColumn("id", monotonically_increasing_id())
##dd.printSchema()
##dd.show()
tokenizer = Tokenizer(inputCol='Document', outputCol='Tokens')
df_words_token = tokenizer.transform(dd).select('id','Document', 'Tokens')
remover = StopWordsRemover(inputCol='Tokens', outputCol='Clean Tokens')
df2 = remover.transform(df_words_token).select('id', 'Clean Tokens')
stemmer = SnowballStemmer(language='english')
stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
df_stemmed = df2.withColumn("words_stemmed", stemmer_udf("Clean Tokens")).select('id', 'words_stemmed')
new_df = dd.join(df_stemmed, on=["id"]).drop("id")
new_df = new_df.withColumn("doc_id", monotonically_increasing_id())
print(new_df.printSchema())
print(new_df.show())
rdd = new_df.rdd
output = rdd.flatMap(lambda line: [(line[0], word) for word in line[1].lower().split(' ')])\
.map(lambda x: (x[1], x[0]))\
.map(lambda x: (x, 1))\
.reduceByKey(lambda x, y: x+y)\
.map(lambda x: (x[0][0], [(x[0][1].replace('file:/Users/maxrojas/Desktop/job search/mondra/Mondra Data-Engineering-Test/dataset/',''))]))\
.reduceByKey(lambda x, y: x+y)
data = output.collect()
print(data)
words_list = new_df.select("words_stemmed").rdd.flatMap(list).collect()
word_list = [x for y in words_list for x in y]
#new_dict = {x: y for y, x in enumerate(reduce_list(words_list))}
new_dict = {x: idx if not x.isdigit() else int(x) for idx, x in enumerate(set(reduce_list(words_list)))}
