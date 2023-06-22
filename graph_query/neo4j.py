from py2neo import Graph

class GraphQuery():
    def __init__(self) -> None:
        self.graph = Graph("bolt://localhost:7687",name=None,auth=("neo4j", "12345678"))
    
    def get_desc(self, entity:str):
        result = self.graph.run(f'MATCH (source)-[rel:has_desc]-(target) WHERE source.name=~"(?i){entity}" RETURN target').data()
        if len(result) == 0:
            return f'未找到{entity}相关的描述😅'
        return result[0]['target']['name']
    
    def get_child_desc(self,parent:str,child:str):
        result = self.graph.run(f'MATCH (p)-[r1:has_child]-(c)-[r2:has_desc]-(t) WHERE p.name=~"(?i){parent}" AND c.name=~"(?i){child}" RETURN t').data()
        if len(result) == 0:
            return f'未找到{parent}中的{child}相关的描述😅'
        return result[0]['t']['name']
        
    def get_children(self,entity:str,child = None):
        if child is None:
            result = self.graph.run(f'MATCH (source)-[rel:has_child]->(t) WHERE source.name=~"(?i){entity}" RETURN t').data()
        else:
            result = self.graph.run(f'MATCH (p)-[r1:has_child]->(c)-[r2:has_child]->(t) WHERE p.name=~"(?i){entity}" AND c.name=~"(?i){child}" RETURN t').data()
        if len(result) == 0:
            if child:
                return f'未找到{entity}中的{child}相关的子知识点😅'
            return f'未找到{entity}相关的子知识点😅'
        if child:
            msg = f'{entity}中的{child}的子知识点有：'
        else:
            msg = f'{entity}的子知识点有：'
        for item in result:
            msg += item['t']['name'] + ','
        return msg[:-1]
    
    def get_next(self,entity:str):
        result = self.graph.run(f'MATCH (m)-[rel:require]->(n) WHERE n.name=~"(?i){entity}" RETURN m').data()
        if len(result) == 0:
            return f'未找到{entity}相关的后置知识点😅'
        msg = f'学完{entity}，可以考虑看看：'
        for item in result:
            msg += item['m']['name'] + ','
        return msg[:-1]
    
    def get_require(self,entity:str):
        result = self.graph.run(f'MATCH (m)-[rel:require]->(n) WHERE m.name=~"(?i){entity}" RETURN n').data()
        if len(result) == 0:
            return f'未找到{entity}相关的前置知识点😅'
        msg = f'学{entity}之前，最好先了解：'
        for item in result:
            msg += item['n']['name'] + ','
        return msg[:-1]
    
    def get_roadmap(self):
        
        start = 'Go语言'
        end = '分布式文件存储'
        result_main = self.graph.run(f'match p = (endn:`知识点`)-[r:require*..10]-(startn:`知识点`) where startn.name="{start}" AND endn.name="{end}" return p').data()
        result_has_child = self.graph.run(f'match (p:`知识点`)-[r:has_child]->(q:`知识点`) return p,q,r').data()
        # result_has_tag = self.graph.run(f'match (p:`文章`)-[r:has_tag]->(q:`知识点`) return p,q,r').data()
        result_entitys = self.graph.run(f'match (p:`知识点`) return p').data()
        
        nodes = []
        rels = []
        for path in result_main:
            for node in path['p'].nodes:
                nodes.append({
                    'name': node['name'],
                    'type': 'main'
                })
            for rel in path['p'].relationships:
                rels.append({
                    'start':rel.start_node['name'],
                    'end':rel.end_node['name'],
                    'type':type(rel).__name__,
                })
        for res in result_has_child:
            if res['r'].end_node['name'] not in [node['name'] for node in nodes]:
                nodes.append({
                    'name': res['r'].end_node['name'],
                    'type' : 'sub'
                })
            rels.append({
                'start':res['r'].start_node['name'],
                'end':res['r'].end_node['name'],
                'type':type(res['r']).__name__,
            })
        # for res in result_has_tag:
        #     if res['r'].start_node['name'] not in [node['name'] for node in nodes]:
        #         nodes.append({
        #             'name': res['r'].start_node['name'],
        #             'type' : 'article'
        #         })
        #     rels.append({
        #         'start':res['r'].start_node['name'],
        #         'end':res['r'].end_node['name'],
        #         'type':type(res['r']).__name__,
            # })
        for entity in result_entitys:
            if entity['p']['name'] not in [node['name'] for node in nodes]:
                nodes.append({
                    'name': entity['p']['name'],
                    'type' : 'sub'
                })
        return {
                'nodes':nodes,
                'rels':rels,
            }
    def get_entity_roadmap(self,entity:str):
        result_has_tag = self.graph.run(f'match (p:`文章`)-[r:has_tag]->(q:`知识点`) where q.name="{entity}" return p,q,r').data()
        print(result_has_tag)
        print(entity)
        nodes = []
        rels = []
        for res in result_has_tag:
            if res['r'].start_node['name'] not in [node['name'] for node in nodes]:
                nodes.append({
                    'name': res['r'].start_node['name'],
                    'type' : 'article'
                })
            rels.append({
                'start':res['r'].start_node['name'],
                'end':res['r'].end_node['name'],
                'type':type(res['r']).__name__,
            })
        return {
                'nodes':nodes,
                'rels':rels,
            }
if __name__ == "__main__":
    neo4j = GraphQuery()
    print(neo4j.get_desc("Vue"))
    