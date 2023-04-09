import json, random, uuid

from py2neo import Graph
from tqdm import tqdm


class KnowledgeGraph(object):

    def __init__(self):
        self.graph = Graph('bolt://localhost:7687', name='neo4j', auth=('neo4j', '12345678'))

        self.nodes = []
        self.k_points = []  # 知识点
        self.k_descriptions = []  # 描述
        self.k_articles = []  # 文章

        self.r_desc = []  # 描述关系
        self.r_require = []  # 前置关系
        self.r_child = []  # 子知识点关系
        self.r_tag = []  # 标签关系

    def extract(self, file_path):
        print('从数据中提取三元组')
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for node in json_data:
            self.__extract_node(json_data[node], node)

    def __extract_node(self, node, node_name, parent_id=None):
        self.k_points.append(node_name)
        node_id = str(self.__create_node(node_name, '知识点', parent_id))
        print(f'写入实体 {node_name},id:{node_id}')
        self.nodes.append(node_name)
        if parent_id:
            self.r_child.append([parent_id, 'has_child', node_id])
        if 'description' in node:
            self.k_descriptions.append(node['description'])
            desc_id = self.__create_node(node['description'], '描述', node_id)
            self.r_desc.append([node_id, 'has_desc', desc_id])
        if 'require' in node:
            for r_item in node['require']:
                r_id = self.__create_node(r_item, '知识点')
                self.r_require.append([node_id, 'require', r_id])
        if 'children' in node:
            for child in node['children']:
                self.__extract_node(node['children'][child], child, node_id)
        if type(node) is str:
            self.k_descriptions.append(node)
            desc_id = self.__create_node(node, '描述', node_id)
            self.r_desc.append([node_id, 'has_desc', desc_id])

    def __create_node(self, name, label, parent=None) -> int:
        if parent is None:
            cql = f'MERGE (n:{label} {{name:"{name}",parent:"ROOT"}}) RETURN id(n) as id'
        else:
            cql = f'MERGE (n:{label} {{name:"{name}",parent:"{parent}"}}) RETURN id(n) as id'
        return self.graph.run(cql).data()[0]['id']

    def write_relations(self, relations, head, tail):
        if len(relations) == 0:
            return
        print(f'写入关系 {relations[0][1]}')
        for relation in tqdm(relations):
            cql = f'MATCH (n:{head}),(m:{tail}) WHERE id(n) = {relation[0]} AND id(m) = {relation[2]} MERGE (n)-[r:{relation[1]}]->(m) RETURN r'
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def clear_database(self):
        print('清空数据库')
        self.graph.delete_all()

    def create_relations(self):
        self.write_relations(self.r_desc, '知识点', '描述')
        self.write_relations(self.r_require, '知识点', '知识点')
        self.write_relations(self.r_child, '知识点', '知识点')
        self.write_relations(self.r_tag, '知识点', '标签')
        
    def extarct_entitys(self):
        with open('./data/entitys.json', 'w', encoding='utf-8') as f:
            json.dump(list(set(self.nodes)), f,ensure_ascii=False, indent=4)


if __name__ == '__main__':
    kg = KnowledgeGraph()
    kg.clear_database()
    kg.extract('./data/data.json')
    kg.create_relations()
    kg.extarct_entitys()