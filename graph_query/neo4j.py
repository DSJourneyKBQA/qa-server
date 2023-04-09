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
if __name__ == "__main__":
    neo4j = GraphQuery()
    print(neo4j.get_desc("Vue"))
    