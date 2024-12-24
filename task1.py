from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, collect_set, size, array_intersect, concat_ws
import sys

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("CommunityDetection") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
# 加载数据并缓存
def load_data(input_path):
    data = spark.read.option("header", "true").option("multiLine", "false")\
    .option("wholeFile", "false").csv(input_path)

    user_business_pairs = data.groupBy("user_id") \
                              .agg(collect_set("business_id").alias("business_ids"))
    return user_business_pairs

# 构建边
def construct_edges(user_business_pairs, threshold):
    user_pairs = user_business_pairs.alias('df1').crossJoin(user_business_pairs.alias('df2')) \
        .filter(F.col('df1.user_id') != F.col('df2.user_id')).distinct()  # 确保每对用户只比较一次
    edges_df = user_pairs.withColumn(
        "common_businesses",
        array_intersect(F.col('df1.business_ids'), F.col('df2.business_ids'))
    ).filter(
        size("common_businesses") >= threshold  # 按阈值过滤，共享商店数大于等于1
    ).select(
        F.col('df1.user_id').alias('src'),
        F.col('df2.user_id').alias('dst'),
        "common_businesses"
    ).distinct()

    

    return edges_df

# 检测社区
def detect_communities(nodes, edges):
    graph = GraphFrame(nodes, edges)
    #sys.exit()
    communities = graph.labelPropagation(maxIter=5)  # 降低最大迭代次数
    return communities

# 保存社区
def save_communities(communities, output_path):
    communities_df = communities.groupBy("label") \
                                .agg(collect_set("id").alias("community")) \
                                .orderBy("community")
    communities_sorted = communities_df.rdd.map(
        lambda row: sorted(row["community"])
    ).sortBy(lambda x: (len(x), x))
    #print("start to save")
    with open(output_path, 'w') as f:
        for community in communities_sorted.collect():
            #print(', '.join("'" + user + "'" for user in community) + '\n')
            f.write(',  '.join("'" + user + "'" for user in community) + '\n')

def main():
    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    
    #filter_threshold = 1
    #input_file_path = "mytest.csv"
    #output_file_path = "myans.txt"
    user_business_pairs = load_data(input_file_path)
    #user_business_pairs.limit(20).show()
    # sys.exit()
    
    edges = construct_edges(user_business_pairs, filter_threshold)
    nodes_src = edges.select(col("src").alias("id"))
    nodes_dst = edges.select(col("dst").alias("id"))
    nodes = nodes_src.union(nodes_dst).distinct()

    #edges.limit(20).show()
    #sys.exit()
    communities = detect_communities(nodes, edges)
    #print("start to save")
    save_communities(communities, output_path=output_file_path)

if __name__ == "__main__":
    main()
    spark.stop()

