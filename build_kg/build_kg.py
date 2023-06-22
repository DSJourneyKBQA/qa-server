import json, random, uuid

from py2neo import Graph
from tqdm import tqdm


class KnowledgeGraph(object):

    def __init__(self):
        self.graph = Graph('bolt://localhost:7687', name='neo4j', auth=('neo4j', '12345678'))

        self.nodes = []
        
        self.n_points = []  # 知识点
        self.n_descriptions = []  # 描述
        self.n_articles = []  # 文章

        self.r_desc = []  # 描述关系
        self.r_require = []  # 前置关系
        self.r_child = []  # 子知识点关系
        self.r_tag = []  # 标签关系

    def extract_data(self, file_path):
        print('从数据中提取三元组')
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for node in json_data:
            self.__extract_node(json_data[node], node)

    def extract_article(self, file_path):
        print('从数据中提取文章')
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for article in json_data:
            self.__extract_article(article)

    def __extract_article(self,data):
        print(f'提取文章 {data["id"]}')
        node_name = f'article_{data["id"]}'
        self.n_articles.append(node_name)
        for tag in data['tags']:
            self.r_tag.append([node_name, 'has_tag', tag])

    def __extract_node(self, node, node_name):
        self.n_points.append(node_name)
        print(f'提取实体 {node_name}')
        self.nodes.append(node_name)
        if 'description' in node:
            self.n_descriptions.append(node['description'])
            self.r_desc.append([node_name, 'has_desc', node['description']])
        if 'require' in node:
            for r_item in node['require']:
                self.r_require.append([node_name, 'require', r_item])
        if 'children' in node:
            for child in node['children']:
                self.r_child.append([node_name, 'has_child', child])
                self.__extract_node(node['children'][child], child)
        if type(node) is str:
            self.n_descriptions.append(node)
            self.r_desc.append([node_name, 'has_desc', node])

    def write_nodes(self, nodes, label):
        '''写入实体'''
        if len(nodes) == 0:
            return
        print(f'写入实体 {label}')
        for node in tqdm(nodes):
            cql = f'MERGE (n:{label} {{name:"{node}"}}) RETURN n'
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def write_relations(self, relations, head, tail):
        '''写入关系'''
        if len(relations) == 0:
            return
        print(f'写入关系 {relations[0][1]}')
        for relation in tqdm(relations):
            cql = f'MATCH (n:{head}),(m:{tail}) WHERE n.name = "{relation[0]}" AND m.name = "{relation[2]}" MERGE (n)-[r:{relation[1]}]->(m) RETURN r'
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def clear_database(self):
        '''清空数据库'''
        print('清空数据库')
        self.graph.delete_all()
        
    def create_nodes(self):
        print(self.n_articles)
        print(self.n_points)
        self.write_nodes(self.n_points, '知识点')
        self.write_nodes(self.n_descriptions, '描述')
        self.write_nodes(self.n_articles, '文章')

    def create_relations(self):
        print(self.r_tag)
        self.write_relations(self.r_desc, '知识点', '描述')
        self.write_relations(self.r_require, '知识点', '知识点')
        self.write_relations(self.r_child, '知识点', '知识点')
        self.write_relations(self.r_tag, '文章', '知识点')

    def extarct_entitys(self):
        with open('./data/entitys.json', 'w', encoding='utf-8') as f:
            json.dump(list(set(self.n_points)), f,ensure_ascii=False, indent=4)


if __name__ == '__main__':
    kg = KnowledgeGraph()
    kg.clear_database()
    kg.extract_data('./data/data.json')
    kg.extract_data('./data/extra.json')
    kg.extract_article('./data/articles.json')
    kg.create_nodes()
    kg.create_relations()
    kg.extarct_entitys()