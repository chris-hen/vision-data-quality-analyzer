# -------------------------------------------------------------------------
# Random Forest decision tree visualization:
# -------------------------------------------------------------------------
#
# This script creates a detailed Graphviz visualization
# of a single decision tree from the trained Random Forest model.
#
# The visualization shows:
#
# - real decision nodes from the trained model
# - feature thresholds used for splitting
# - prediction flow through the tree
# - final leaf predictions
# - collapsed subtrees for large branches
#
# The generated visualization is mainly used for:
#
# - dashboard visualization
# - model explanation
# - Random Forest interpretation
# - presentation and documentation
# -------------------------------------------------------------------------


import os
import joblib
import graphviz


# path to trained Random Forest model
MODEL_PATH = "models/random_forest_model.pkl"

# output folder for generated visualization
VISUALIZATION_DIR = "results/visualizations"

# create output folder if it does not exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


# ------------------------------------------------------------
# Load trained Random Forest model
# ------------------------------------------------------------

# load trained RF classifier
model = joblib.load(MODEL_PATH)

# extracted OpenCV feature names used by the RF model
features = ["blur_score", "brightness", "contrast"]

# class names learned by the RF classifier
classes = list(model.classes_)

# selected decision tree visualized from the ensemble
TREE_INDEX = 67

# maximum displayed visualization depth
MAX_DEPTH_SHOWN = 3

# extract selected decision tree
estimator = model.estimators_[TREE_INDEX]
tree = estimator.tree_


# ------------------------------------------------------------
# Color palette
# ------------------------------------------------------------

# colors for input feature node
INPUT_FILL = "#332B14"
INPUT_BORDER = "#FFD166"
INPUT_EDGE = "#4EA8FF"

# colors for decision tree nodes
TREE_FILL = "#14213D"
TREE_BORDER = "#4EA8FF"
TREE_EDGE = "#4EA8FF"

# color for FALSE branch edges
FALSE_EDGE = "#FF6B8A"

# colors for collapsed subtree nodes
SUBTREE_FILL = "#2C2452"
SUBTREE_BORDER = "#A78BFF"

# colors for final prediction leaf nodes
FINAL_FILL = "#0F3D2E"
FINAL_BORDER = "#4DFFB8"
FINAL_EDGE = "#4DFFB8"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

# return predicted class for a given node
def get_prediction(node_id):

    class_id = tree.value[node_id][0].argmax()

    return classes[class_id]


# create formatted Graphviz label for tree nodes
def node_label(node_id, depth):

    feature_id = tree.feature[node_id]
    samples = int(tree.n_node_samples[node_id])

    prediction = get_prediction(node_id)

    # final leaf node
    if feature_id == -2:

        return (
            f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='4'>"
            f"<TR><TD><FONT POINT-SIZE='14'>FINAL LEAF</FONT></TD></TR>"
            f"<TR><TD><FONT POINT-SIZE='24'><B>{prediction}</B></FONT></TD></TR>"
            f"<TR><TD><FONT POINT-SIZE='13'>samples: {samples}</FONT></TD></TR>"
            f"</TABLE>>"
        )

    # collapsed subtree visualization
    if depth >= MAX_DEPTH_SHOWN:

        return (
            f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='4'>"
            f"<TR><TD><FONT POINT-SIZE='14'>SUBTREE</FONT></TD></TR>"
            f"<TR><TD><FONT POINT-SIZE='22'><B>{prediction}</B></FONT></TD></TR>"
            f"<TR><TD><FONT POINT-SIZE='13'>tree continues below</FONT></TD></TR>"
            f"<TR><TD><FONT POINT-SIZE='13'>samples: {samples}</FONT></TD></TR>"
            f"</TABLE>>"
        )

    # extract decision feature and threshold
    feature_name = features[feature_id]
    threshold = tree.threshold[node_id]

    # regular decision node
    return (
        f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='4'>"
        f"<TR><TD><FONT POINT-SIZE='14'>DECISION NODE</FONT></TD></TR>"
        f"<TR><TD><FONT POINT-SIZE='21'><B>{feature_name}</B></FONT></TD></TR>"
        f"<TR><TD><FONT POINT-SIZE='25'><B>≤ {threshold:.2f}</B></FONT></TD></TR>"
        f"<TR><TD><FONT POINT-SIZE='13'>samples: {samples}</FONT></TD></TR>"
        f"</TABLE>>"
    )


# recursively create Graphviz nodes and edges
def add_nodes(graph, node_id, depth=0):

    feature_id = tree.feature[node_id]

    # check if node is a real leaf
    is_real_leaf = feature_id == -2

    # collapse subtree if visualization depth is exceeded
    is_depth_limit = depth >= MAX_DEPTH_SHOWN and not is_real_leaf

    # create final prediction leaf
    if is_real_leaf:

        graph.node(
            str(node_id),
            label=node_label(node_id, depth),
            fillcolor=FINAL_FILL,
            color=FINAL_BORDER,
            penwidth="2.5"
        )

        return

    # create collapsed subtree node
    if is_depth_limit:

        graph.node(
            str(node_id),
            label=node_label(node_id, depth),
            style="rounded,dashed,filled",
            fillcolor=SUBTREE_FILL,
            color=SUBTREE_BORDER,
            penwidth="2.3"
        )

        return

    # create regular decision node
    graph.node(
        str(node_id),
        label=node_label(node_id, depth),
        fillcolor=TREE_FILL,
        color=TREE_BORDER,
        penwidth="2.5"
    )

    # extract child node ids
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    # recursively create child nodes
    add_nodes(graph, left_child, depth + 1)
    add_nodes(graph, right_child, depth + 1)

    # connect TRUE branch
    graph.edge(
        str(node_id),
        str(left_child),
        label=" TRUE ",
        color=FINAL_EDGE,
        fontcolor=FINAL_EDGE,
        penwidth="2.3",
        fontsize="13",
        labeldistance="2.8",
        labelangle="-30"
    )

    # connect FALSE branch
    graph.edge(
        str(node_id),
        str(right_child),
        label=" FALSE ",
        color=FALSE_EDGE,
        fontcolor=FALSE_EDGE,
        penwidth="2.3",
        fontsize="13",
        labeldistance="2.8",
        labelangle="30"
    )


# ------------------------------------------------------------
# Create Graphviz diagram
# ------------------------------------------------------------

# initialize Graphviz directed graph
graph = graphviz.Digraph(
    name=f"Random Forest Tree {TREE_INDEX}",
    format="svg"
)

# configure global graph appearance
graph.attr(
    rankdir="TB",
    bgcolor="#070B16",
    splines="ortho",
    nodesep="0.75",
    ranksep="0.9",
    pad="0.35",
    margin="0.10",
    fontname="Arial",
    fontcolor="#F5F7FB"
)

# configure global node appearance
graph.attr(
    "node",
    shape="box",
    style="rounded,filled",
    fontname="Arial",
    fontcolor="#F5F7FB",
    margin="0.20,0.14"
)

# configure global edge appearance
graph.attr(
    "edge",
    fontname="Arial",
    arrowsize="0.75"
)


# ------------------------------------------------------------
# Create decision tree structure
# ------------------------------------------------------------

# recursively generate tree visualization
add_nodes(graph, node_id=0)


# ------------------------------------------------------------
# Title node
# ------------------------------------------------------------

# create visualization title node
graph.node(
    "title_node",
    label=(
        f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='9'>"

        f"<TR><TD ALIGN='LEFT'>"
        f"<FONT POINT-SIZE='42'><B>"
        f"Random Forest Tree #{TREE_INDEX}"
        f"</B></FONT>"
        f"</TD></TR>"

        f"<TR><TD ALIGN='LEFT'>"
        f"<FONT POINT-SIZE='22'>"
        f"Real decision structure from the trained model"
        f"</FONT>"
        f"</TD></TR>"

        f"<TR><TD ALIGN='LEFT'>"
        f"<FONT POINT-SIZE='18'>"
        f"Displayed depth: {MAX_DEPTH_SHOWN} · "
        f"Actual depth: {tree.max_depth} · "
        f"Total nodes: {tree.node_count}"
        f"</FONT>"
        f"</TD></TR>"

        f"<TR><TD ALIGN='LEFT'>"
        f"<FONT POINT-SIZE='17'>"
        f"Blue: decision node · Green: final leaf · "
        f"Purple: collapsed subtree"
        f"</FONT>"
        f"</TD></TR>"

        f"</TABLE>>"
    ),

    shape="plain",
    style="",
    fontcolor="#F5F7FB",
    margin="0"
)


# ------------------------------------------------------------
# Input feature node
# ------------------------------------------------------------

# create input feature node
graph.node(
    "input_node",
    label=(
        f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='6'>"

        f"<TR><TD>"
        f"<FONT POINT-SIZE='14'>INPUT</FONT>"
        f"</TD></TR>"

        f"<TR><TD>"
        f"<FONT POINT-SIZE='26'><B>Image Features</B></FONT>"
        f"</TD></TR>"

        f"<TR><TD>"
        f"<FONT POINT-SIZE='19'>"
        f"blur_score · brightness · contrast"
        f"</FONT>"
        f"</TD></TR>"

        f"</TABLE>>"
    ),

    fillcolor=INPUT_FILL,
    color=INPUT_BORDER,
    penwidth="2.8",

    margin="0.32,0.24"
)


# ------------------------------------------------------------
# Layout control
# ------------------------------------------------------------

# place title node on same level as root node
with graph.subgraph() as title_rank:

    title_rank.attr(rank="same")
    title_rank.node("title_node")
    title_rank.node("0")

# invisible layout edge for positioning
graph.edge(
    "title_node",
    "0",
    style="invis",
    weight="10",
    minlen="1"
)

# place input node near root node
with graph.subgraph() as input_rank:

    input_rank.attr(rank="same")
    input_rank.node("0")
    input_rank.node("input_node")

# invisible layout edge for positioning
graph.edge(
    "0",
    "input_node",
    style="invis",
    weight="10",
    minlen="2"
)

# connect input feature node to tree root
graph.edge(
    "input_node",
    "0",
    label=" feature vector ",
    color=INPUT_BORDER,
    fontcolor=INPUT_BORDER,
    penwidth="3.0",
    arrowsize="1.0",
    fontsize="12"
)


# ------------------------------------------------------------
# Export visualization
# ------------------------------------------------------------

# render final Graphviz visualization
graph.render(
    f"{VISUALIZATION_DIR}/random_forest_tree_example_from_model",
    cleanup=True
)

# print summary
print(f"Tree {TREE_INDEX} visualization saved.")