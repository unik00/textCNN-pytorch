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


def find(node, original_position):
    """ Find first object that has text s.
    Params:
        original_position: an integer
    Returns:
        node: spacy node
    """
    if node.idx == original_position:
        return node
    for child in node.children:
        t = find(child, original_position)
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

    
def get_shortest_path(en_nlp_docs, sentence, e1_position, e2_position, original_text):
    """ Find the shortest path between given pair of entities.
        Returns both words and POS tags
    Args:
        en_nlp_docs: spacy docs
        sentence: str
        e1_position: str, start entity
        e2_position: str, end entity
        original_text: for debugging only
    Returns:
        path: pair of strings (word, POS tag)
    """
    # if len(start.split(' ')) > 1:
    #     start = start.split(' ')[0]
    # if len(end.split(' ')) > 1:
    #     end = end.split(' ')[0]
    #
    # if len(start.split('-')) > 1:
    #     start = start.split('-')[0]
    # if len(end.split('-')) > 1:
    #     end = end.split('-')[0]

    doc = en_nlp_docs
    for sent in doc.sents:
        start_node = find(sent.root, e1_position)
        end_node = find(sent.root, e2_position)
        # print(start_node, end_node, e1_position, e2_position)
        # print(original_text)
        if start_node and end_node:
            if (start_node.idx != e1_position) or (end_node.idx != e2_position):
                print("wwrong")
                print("startnode idx: ", start_node.idx)
                print(sentence)
                print(e1_position)
                print(original_text)
            # assert start_node.idx == e1_position

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
        path = [(end_node.orth_, end_node.pos_, end_node.dep_)]
        while end_node != start_node:
            end_node = trace[end_node]
            #print(end_node.orth_, end_node.pos_)
            path.append((end_node.orth_, end_node.pos_, end_node.dep_))
        path = path[::-1]
        return path
    print("Cannot parse \"{}\", returning empty array.".format(sentence))
    return []


if __name__ == "__main__":
    s = "They tried an assault of their own an hour later, with two columns of sixteen tanks backed by a battalion of Panzer grenadiers"
    s = "He removed the glass slide precleaned in piranha solution that was placed upright in a beaker"
    print_dependency_tree(s)
    print(get_shortest_path(s, "slide", "beaker"))
