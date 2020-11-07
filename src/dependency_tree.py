import spacy

en_nlp = spacy.load("en_core_web_sm")


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
            trace[v] = (u, [0, 1])
            dfs(v, e, trace)

    if u.head and u.head not in trace:
        trace[u.head] = (u, [1, 0])
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

        start_node_token_index = start_node.i
        end_node_token_index = end_node.i

        dfs(start_node, end_node, trace)
        # print(sent, "---", start,"---", end)

        path = [(end_node.orth_,
                 end_node.pos_,
                 end_node.dep_,
                 [0, 0], # no edge here
                 start_node_token_index-end_node.i,
                 end_node_token_index-end_node.i)]

        while end_node != start_node:
            end_node, edge_direction = trace[end_node]
            #print(end_node.orth_, end_node.pos_)
            path.append((end_node.orth_,
                         end_node.pos_,
                         end_node.dep_,
                         edge_direction,
                         start_node_token_index - end_node.i,
                         end_node_token_index - end_node.i
                         ))
        path = path[::-1]
        return path
    print("Cannot parse \"{}\", returning empty array.".format(sentence))
    return []


'''
def get_shortest_path(en_nlp_docs, sentence, e1_position, e2_position, original_text):
    doc = en_nlp_docs
    path = []

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

        start_node_token_index = start_node.i
        end_node_token_index = end_node.i

        for end_node in sent:
            #print(end_node.orth_, end_node.pos_)
            path.append((end_node.orth_,
                         end_node.pos_,
                         end_node.dep_,
                         start_node_token_index - end_node.i,
                         end_node_token_index - end_node.i
                         ))
        # path = path[::-1]
        return path
    print("Cannot parse \"{}\", returning empty array.".format(sentence))
    return []

'''

if __name__ == "__main__":
    s = "They tried an assault of their own an hour later, with two columns of sixteen tanks backed by a battalion of Panzer grenadiers"
    s = "He removed the glass slide precleaned in piranha solution that was placed upright in a beaker"
    print_dependency_tree(s)
