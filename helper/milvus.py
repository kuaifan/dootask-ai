from pymilvus import MilvusClient
from langchain_community.embeddings import DashScopeEmbeddings
from typing import List, Dict, Optional

class FoodSearchTool:
    def __init__(self, db_path: str = "./milvus_demo.db", collection_name: str = "food_data"):
        """
        初始化食品搜索工具
        
        :param db_path: Milvus数据库路径
        :param collection_name: 集合名称
        """
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        # 阿里云的向量提取，可更换为其他的
        self.embedding_fn = DashScopeEmbeddings(
            model="text-embedding-v3", 
            dashscope_api_key=""
        )
        self.initialize_collection(dimension=1024)
        
    def initialize_collection(self, dimension: int = 1024):
        """
        初始化集合
        
        :param dimension: 向量维度
        """
        if not self.client.has_collection(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=dimension
            )
    
    def insert_data(self, data_list: List[Dict]) -> Dict:
        """
        插入数据到Milvus
        
        :param data_list: 要插入的数据列表，每个元素应包含name和url字段
        :return: 插入结果
        """
        names = [item["name"] for item in data_list]
        vectors = self.embedding_fn.embed_documents(names)
        
        entities = []
        for i, data in enumerate(data_list):
            entity = {
                "id": i,
                "name": data["name"],
                "url": data["url"],
                "vector": vectors[i],
            }
            entities.append(entity)

        res = self.client.insert(collection_name=self.collection_name, data=entities)
        return res
    
    def search_by_name(self, query_names: List[str], top_k: int = 3, threshold: float = 0.75) -> List[Dict]:
        """
        通过名称搜索相似菜品
        
        :param query_names: 查询名称列表
        :param top_k: 返回每个查询的最相似结果数量
        :param threshold: 相似度阈值，只返回大于此阈值的结果
        :return: 搜索结果列表
        """
        query_vector = self.embedding_fn.embed_documents(query_names)
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=query_vector,
            limit=top_k,
            output_fields=["name", "url"],
            search_params={"metric_type": "COSINE"}
        )
        
        formatted_results = []
        for i, query in enumerate(query_names):
            query_result = results[i]
            best_match = None
            best_score = threshold  # 初始值为阈值，只有超过它的才会被记录
            for item in query_result:
                current_score = item.get("distance", 0)
                if current_score > best_score:
                    best_score = current_score
                    best_match = item
            
            if best_match is not None:
                formatted_results.append({
                    "query": query,
                    "name": best_match["entity"]["name"],
                    "url": best_match["entity"]["url"],
                    "score": best_score
                })
            else:
                formatted_results.append({
                    "query": query,
                    "name": None,
                    "url": None,
                    "score": None
                })
        return formatted_results
    
    def close(self):
        """关闭客户端连接"""
        self.client.close()


# 示例用法
if __name__ == "__main__":
    # 示例数据
    sample_data = [
        {"name": "番茄炒蛋", "url": "http://example.com/tomato_egg"},
        {"name": "西红柿炒鸡蛋", "url": "http://example.com/tomato_egg2"},
        {"name": "红烧肉", "url": "http://example.com/braised_pork"},
        {"name": "糖醋排骨", "url": "http://example.com/sweet_pork_ribs"},
        {"name": "蛋炒饭", "url": "http://example.com/egg_fried_rice"},
        {"name": "番茄牛肉", "url": "http://example.com/tomato_beef"},
        {"name": "西红柿牛腩", "url": "http://example.com/tomato_beef_brisket"},
        {"name": "炒鸡蛋", "url": "http://example.com/fried_egg"},
    ]
    
    # 初始化工具
    search_tool = FoodSearchTool()
    
    # 如果集合不存在，则初始化并插入数据
    if search_tool.collection_name not in search_tool.client.list_collections():
        print("初始化集合并插入数据...")
        search_tool.initialize_collection()
        search_tool.insert_data(sample_data)
    else:
        print("集合已存在，直接查询")
    
    # 测试搜索
    queries = ["番茄鸡蛋", "东坡肉"]
    print(f"\n搜索查询: {queries}")
    results = search_tool.search_by_name(queries)
    
    # 打印结果
    for result in results:
        print(f"名称: {result['name']}, URL: {result['url']}, 相似度: {result['score']:.4f}")
    
    # 关闭连接
    search_tool.close()