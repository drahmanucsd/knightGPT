import networkx as nx

class GraphStore:
    """
    A simple graph wrapper using NetworkX for storing Publication nodes and edges.
    """
    def __init__(self):
        # Initialize an undirected graph
        self.G = nx.Graph()

    def add_publication(self, pub_id, title, authors, year, abstract):
        import json
        # authors list → JSON string
        authors_str = json.dumps(authors)
        # sanitize year (no NoneTypes)
        year_val = year if year is not None else 0
        self.G.add_node(
            pub_id,
            type="Publication",
            title=title or "",
            authors=authors_str,
            year=year_val,
            abstract=abstract or ""
        )

    def add_edge(self, source_id: str, target_id: str, **attrs):
        """
        Add an edge between two nodes, with optional attributes.
        Useful later for 'cites', 'mentions', etc.

        Args:
            source_id (str): ID of the source node.
            target_id (str): ID of the target node.
            attrs: Additional edge metadata (e.g. weight, type).
        """
        self.G.add_edge(source_id, target_id, **attrs)

    def save(self, path: str):
        """
        Persist the graph to disk in GEXF format.

        Args:
            path (str): File path to write the graph (e.g. 'graph.gexf').
        """
        nx.write_gexf(self.G, path)

    def load(self, path: str):
        """
        Load a graph from a GEXF file.

        Args:
            path (str): File path of the saved graph.
        """
        self.G = nx.read_gexf(path)

    def get_publication(self, pub_id: str) -> dict:
        """
        Retrieve a Publication node's data by its ID.

        Args:
            pub_id (str): The publication ID.

        Returns:
            dict: Node attributes if found, otherwise None.
        """
        data = self.G.nodes.get(pub_id)
        return data if data and data.get('type') == 'Publication' else None

    def list_publications(self) -> list:
        """
        List all Publication node IDs in the graph.

        Returns:
            list of str: Publication IDs.
        """
        return [n for n, d in self.G.nodes(data=True) if d.get('type') == 'Publication']
