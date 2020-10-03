import spacy
from nltk import Tree

en_nlp = spacy.load("en_core_web_sm")


def to_nltk_tree(node):
    # print(node.head)
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def print_dependency_tree(s):
    """
    Params:
        s: string
    """
    doc = en_nlp(s)
    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


def find(node, s):
    """ Find first object that has text s.
    Params:
        s: string
    Returns:
        node: spacy node
    """
    if node.orth_ == s:
        return node
    for child in node.children:
        t = find(child, s)
        if t is not None:
            return t
    return None


def dfs(u, e, trace):
    """
    Args:
        u, e, cur : node
            u is current node
            e is end node
            current is the current node.
        trace: a dict to track visited nodes.
    
    Returns:
        s: string
    """
    if u == e:
        return

    for v in u.children:
        if v not in trace:
            trace[v] = u
            dfs(v, e, trace)

    if u.head and u.head not in trace:
        trace[u.head] = u
        dfs(u.head, e, trace)

    
def get_shortest_path(sentence, start, end):
    """ Find the shortest path between given pair of entities.
    Args:
        sentence: str
        start: str, start entity
        end: str, end entity
    Returns:
        path: string
    """
    if len(start.split(' ')) > 1:
        start = start.split(' ')[0]
    if len(end.split(' ')) > 1:
        end = end.split(' ')[0]

    if len(start.split('-')) > 1:
        start = start.split('-')[0]
    if len(end.split('-')) > 1:
        end = end.split('-')[0]

    doc = en_nlp(sentence)
    for sent in doc.sents:
        start_node = find(sent.root, start)
        end_node = find(sent.root, end)
        
        if start_node is None:
            # print("Cannot find \"{}\" in \"{}\"".format(start, str(sent)))
            # print_dependency_tree(sentence)
            continue
        if end_node is None:
            # print("Cannot find \"{}\" in \"{}\"".format(end, str(sent)))
            # print_dependency_tree(sentence)
            continue

        trace = dict()
        dfs(start_node, end_node, trace)
        # print(sent, "---", start,"---", end)
        path = [end_node.orth_]
        while end_node != start_node:
            end_node = trace[end_node]
            path.append(end_node.orth_)
        path = path[::-1]
        return " ".join(path)
    print("Cannot parse \"{}\", returning empty string.".format(sentence))
    return ""


if __name__ == "__main__":
    s = "They tried an assault of their own an hour later, with two columns of sixteen tanks backed by a battalion of Panzer grenadiers"
    print_dependency_tree(s)
    print(get_shortest_path(s, "battalion", "grenadiers"))
