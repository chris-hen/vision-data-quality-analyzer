# -------------------------------------------------------------------------
# Random Forest process visualization:
# -------------------------------------------------------------------------
#
# This script creates a custom Graphviz visualization of the
# Random Forest classification process used in the dashboard.
#
# The visualization illustrates:
#
# - extracted image quality features
# - multiple decision trees
# - individual tree predictions
# - majority voting across the ensemble
# - final degradation classification
# - example voting behavior
#
# The generated visualization is mainly used for:
#
# - dashboard visualization
# - project presentation
# - Random Forest explanation
# - model architecture overview
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

# total number of decision trees inside the RF ensemble
N_TREES = len(model.estimators_)

# selected trees visualized individually in dashboard graphic
TREE_INDICES = [0, 1, 2, 3]

# remaining trees summarized as compact placeholder node
OTHER_TREES = N_TREES - len(TREE_INDICES)

# collect tree statistics from trained RF ensemble
tree_depths = [estimator.tree_.max_depth for estimator in model.estimators_]
tree_nodes = [estimator.tree_.node_count for estimator in model.estimators_]

# calculate average RF tree statistics
avg_depth = sum(tree_depths) / len(tree_depths)
avg_nodes = sum(tree_nodes) / len(tree_nodes)


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

# colors for prediction nodes
PREDICTION_FILL = "#11354F"
PREDICTION_BORDER = "#5EEBFF"
PREDICTION_EDGE = "#5EEBFF"

# colors for voting nodes
VOTE_FILL = "#2C2452"
VOTE_BORDER = "#A78BFF"
VOTE_EDGE = "#A78BFF"

# colors for example vote visualization
EXAMPLE_VOTE_FILL = "#352E5F"
EXAMPLE_VOTE_BORDER = "#9B8CDB"

# colors for final prediction output
FINAL_FILL = "#0F3D2E"
FINAL_BORDER = "#4DFFB8"
FINAL_EDGE = "#4DFFB8"

# colors for example output node
EXAMPLE_FINAL_FILL = "#18382D"
EXAMPLE_FINAL_BORDER = "#6BCF9E"


# ------------------------------------------------------------
# Create Graphviz diagram
# ------------------------------------------------------------

# initialize Graphviz directed graph
graph = graphviz.Digraph(
    name="Random Forest Dashboard Process",
    format="svg"
)

# configure global graph appearance
graph.attr(
    rankdir="TB",
    bgcolor="#070B16",
    splines="ortho",
    nodesep="0.65",
    ranksep="0.85",
    pad="0.5",
    margin="0.10",
    fontname="Arial",
    fontcolor="#F5F7FB",

    # dashboard visualization title
    label=(
        f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='6'>"
        f"<TR><TD><FONT POINT-SIZE='30'><B>Random Forest Classification Process</B></FONT></TD></TR>"
        f"<TR><TD><FONT POINT-SIZE='15'>"
        f"{N_TREES} decision trees classify image quality metrics independently"
        f"</FONT></TD></TR>"
        f"<TR><TD><FONT POINT-SIZE='11'>"
        f"Average depth: {avg_depth:.1f} · Average nodes: {avg_nodes:.1f}"
        f"</FONT></TD></TR>"
        f"</TABLE>>"
    ),
    labelloc="t"
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
# Input feature node
# ------------------------------------------------------------

# create input feature node
graph.node(
    "input",
    label=(
        "<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='4'>"
        "<TR><TD><FONT POINT-SIZE='12'>INPUT</FONT></TD></TR>"
        "<TR><TD><FONT POINT-SIZE='22'><B>Image Features</B></FONT></TD></TR>"
        "<TR><TD><FONT POINT-SIZE='11'>blur_score · brightness · contrast</FONT></TD></TR>"
        "</TABLE>>"
    ),
    fillcolor=INPUT_FILL,
    color=INPUT_BORDER,
    penwidth="2.5"
)


# ------------------------------------------------------------
# Decision tree layer
# ------------------------------------------------------------

# store all generated tree node ids
tree_node_ids = []

# create individual visualization nodes for selected trees
for idx in TREE_INDICES:

    # extract statistics for current tree
    depth = model.estimators_[idx].tree_.max_depth
    nodes = model.estimators_[idx].tree_.node_count

    tree_id = f"tree_{idx}"
    tree_node_ids.append(tree_id)

    # create decision tree node
    graph.node(
        tree_id,
        label=(
            f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='3'>"
            f"<TR><TD><FONT POINT-SIZE='11'>TREE #{idx}</FONT></TD></TR>"
            f"<TR><TD><FONT POINT-SIZE='18'><B>Decision Tree</B></FONT></TD></TR>"
            f"<TR><TD><FONT POINT-SIZE='10'>depth: {depth}</FONT></TD></TR>"
            f"<TR><TD><FONT POINT-SIZE='10'>nodes: {nodes}</FONT></TD></TR>"
            f"</TABLE>>"
        ),
        fillcolor=TREE_FILL,
        color=TREE_BORDER,
        penwidth="2.2"
    )

# placeholder node for all remaining trees
graph.node(
    "other_trees",
    label=(
        f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='3'>"
        f"<TR><TD><FONT POINT-SIZE='22'><B>...</B></FONT></TD></TR>"
        f"<TR><TD><FONT POINT-SIZE='14'><B>{OTHER_TREES} other trees</B></FONT></TD></TR>"
        f"<TR><TD><FONT POINT-SIZE='10'>not shown individually</FONT></TD></TR>"
        f"</TABLE>>"
    ),
    style="rounded,dashed,filled",
    fillcolor=TREE_FILL,
    color=TREE_BORDER,
    penwidth="2"
)

tree_node_ids.append("other_trees")

# force all tree nodes into same visualization row
with graph.subgraph() as rank_trees:

    rank_trees.attr(rank="same")

    for node_id in tree_node_ids:
        rank_trees.node(node_id)


# ------------------------------------------------------------
# Prediction layer
# ------------------------------------------------------------

# store all prediction node ids
prediction_node_ids = []

# create prediction node for every visualized tree
for idx in TREE_INDICES:

    pred_id = f"pred_{idx}"
    prediction_node_ids.append(pred_id)

    graph.node(
        pred_id,
        label=(
            f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='2'>"
            f"<TR><TD><FONT POINT-SIZE='10'>TREE #{idx}</FONT></TD></TR>"
            f"<TR><TD><FONT POINT-SIZE='15'><B>Class Prediction</B></FONT></TD></TR>"
            f"</TABLE>>"
        ),
        fillcolor=PREDICTION_FILL,
        color=PREDICTION_BORDER,
        penwidth="1.8"
    )

# placeholder prediction node for all remaining trees
graph.node(
    "pred_other",
    label=(
        f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='2'>"
        f"<TR><TD><FONT POINT-SIZE='10'>OTHER TREES</FONT></TD></TR>"
        f"<TR><TD><FONT POINT-SIZE='15'><B>Class Predictions</B></FONT></TD></TR>"
        f"</TABLE>>"
    ),
    style="rounded,dashed,filled",
    fillcolor=PREDICTION_FILL,
    color=PREDICTION_BORDER,
    penwidth="1.8"
)

prediction_node_ids.append("pred_other")

# force prediction nodes into same visualization row
with graph.subgraph() as rank_predictions:

    rank_predictions.attr(rank="same")

    for node_id in prediction_node_ids:
        rank_predictions.node(node_id)


# ------------------------------------------------------------
# Majority voting and final prediction
# ------------------------------------------------------------

# majority voting node
graph.node(
    "vote",
    label=(
        "<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='4'>"
        "<TR><TD><FONT POINT-SIZE='12'>ENSEMBLE AGGREGATION</FONT></TD></TR>"
        "<TR><TD><FONT POINT-SIZE='24'><B>Majority Voting</B></FONT></TD></TR>"
        "<TR><TD><FONT POINT-SIZE='11'>most frequent class becomes final output</FONT></TD></TR>"
        "</TABLE>>"
    ),
    fillcolor=VOTE_FILL,
    color=VOTE_BORDER,
    penwidth="2.4"
)

# final RF prediction output node
graph.node(
    "final_decision",
    label=(
        "<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='4'>"
        "<TR><TD><FONT POINT-SIZE='12'>OUTPUT</FONT></TD></TR>"
        "<TR><TD><FONT POINT-SIZE='26'><B>Degradation Type</B></FONT></TD></TR>"
        "<TR><TD><FONT POINT-SIZE='10'>"
        "original · noise · lowcontrast · dark · blur · bright"
        "</FONT></TD></TR>"
        "</TABLE>>"
    ),
    fillcolor=FINAL_FILL,
    color=FINAL_BORDER,
    penwidth="2.8"
)


# ------------------------------------------------------------
# Example voting visualization
# ------------------------------------------------------------

# example vote distribution across all trees
example_votes = {
    "original": 6,
    "noise": 9,
    "lowcontrast": 71,
    "dark": 0,
    "blur": 4,
    "bright": 10
}

# calculate total votes
total_votes = sum(example_votes.values())

# determine winning class
predicted_class = max(example_votes, key=example_votes.get)

# calculate example confidence value
prediction_confidence = (
    example_votes[predicted_class] / total_votes
) * 100

# example vote table node
graph.node(
    "vote_example",
    label=(
        f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='5'>"

        f"<TR><TD COLSPAN='11'>"
        f"<FONT POINT-SIZE='12'><B>Example Vote ({N_TREES} Trees)</B></FONT>"
        f"</TD></TR>"

        f"<TR>"

        f"<TD ALIGN='RIGHT'><FONT POINT-SIZE='11'>original</FONT></TD>"
        f"<TD ALIGN='LEFT'><FONT POINT-SIZE='11'><B>{example_votes['original']}</B></FONT></TD>"
        f"<TD WIDTH='12'></TD>"

        f"<TD ALIGN='RIGHT'><FONT POINT-SIZE='11'>noise</FONT></TD>"
        f"<TD ALIGN='LEFT'><FONT POINT-SIZE='11'><B>{example_votes['noise']}</B></FONT></TD>"
        f"<TD WIDTH='12'></TD>"

        f"<TD ALIGN='RIGHT'><FONT POINT-SIZE='11'>lowcontrast</FONT></TD>"
        f"<TD ALIGN='LEFT'><FONT POINT-SIZE='11'><B>{example_votes['lowcontrast']}</B></FONT></TD>"

        f"</TR>"

        f"<TR>"

        f"<TD ALIGN='RIGHT'><FONT POINT-SIZE='11'>dark</FONT></TD>"
        f"<TD ALIGN='LEFT'><FONT POINT-SIZE='11'><B>{example_votes['dark']}</B></FONT></TD>"
        f"<TD WIDTH='12'></TD>"

        f"<TD ALIGN='RIGHT'><FONT POINT-SIZE='11'>blur</FONT></TD>"
        f"<TD ALIGN='LEFT'><FONT POINT-SIZE='11'><B>{example_votes['blur']}</B></FONT></TD>"
        f"<TD WIDTH='12'></TD>"

        f"<TD ALIGN='RIGHT'><FONT POINT-SIZE='11'>bright</FONT></TD>"
        f"<TD ALIGN='LEFT'><FONT POINT-SIZE='11'><B>{example_votes['bright']}</B></FONT></TD>"

        f"</TR>"

        f"</TABLE>>"
    ),

    fillcolor=EXAMPLE_VOTE_FILL,
    color=EXAMPLE_VOTE_BORDER,
    penwidth="1.6"
)

# example output class
final_class = predicted_class

# example final prediction node
graph.node(
    "example_class",
    label=(
        f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='4'>"

        f"<TR><TD>"
        f"<FONT POINT-SIZE='12'>EXAMPLE OUTPUT</FONT>"
        f"</TD></TR>"

        f"<TR><TD>"
        f"<FONT POINT-SIZE='25'><B>{final_class}</B></FONT>"
        f"</TD></TR>"

        f"<TR><TD>"
        f"<FONT POINT-SIZE='11'>"
        f"prediction confidence: <B>{prediction_confidence:.1f}%</B>"
        f"</FONT>"
        f"</TD></TR>"

        f"<TR><TD>"
        f"<FONT POINT-SIZE='10'>"
        f"based on example voting"
        f"</FONT>"
        f"</TD></TR>"

        f"</TABLE>>"
    ),

    fillcolor=EXAMPLE_FINAL_FILL,
    color=EXAMPLE_FINAL_BORDER,
    penwidth="1.8"
)

# force voting nodes into same row
with graph.subgraph() as rank_vote:

    rank_vote.attr(rank="same")
    rank_vote.node("vote")
    rank_vote.node("vote_example")

# force output nodes into same row
with graph.subgraph() as rank_final:

    rank_final.attr(rank="same")
    rank_final.node("final_decision")
    rank_final.node("example_class")


# ------------------------------------------------------------
# Create graph connections
# ------------------------------------------------------------

# connect input features to decision trees
for tree_id in tree_node_ids:

    graph.edge(
        "input",
        tree_id,
        color=INPUT_EDGE,
        penwidth="1.6"
    )

# connect decision trees to prediction nodes
for idx in TREE_INDICES:

    graph.edge(
        f"tree_{idx}",
        f"pred_{idx}",
        color=PREDICTION_EDGE,
        penwidth="1.5"
    )

# connect remaining trees to combined prediction node
graph.edge(
    "other_trees",
    "pred_other",
    color=PREDICTION_EDGE,
    penwidth="1.5"
)

# connect all predictions to majority voting
for pred_id in prediction_node_ids:

    graph.edge(
        pred_id,
        "vote",
        color=VOTE_EDGE,
        penwidth="1.6"
    )

# connect majority vote to final prediction
graph.edge(
    "vote",
    "final_decision",
    color=FINAL_EDGE,
    penwidth="2.8",
    arrowsize="0.9"
)

# connect majority voting to example voting explanation
graph.edge(
    "vote",
    "vote_example",
    color=EXAMPLE_VOTE_BORDER,
    penwidth="1.4",
    style="dashed",
    arrowhead="none"
)

# connect example vote to example output
graph.edge(
    "vote_example",
    "example_class",
    color=EXAMPLE_FINAL_BORDER,
    penwidth="1.8",
    arrowsize="0.75"
)


# ------------------------------------------------------------
# Export visualization
# ------------------------------------------------------------

# render final Graphviz visualization
graph.render(
    f"{VISUALIZATION_DIR}/random_forest_concept___",
    cleanup=True
)

# print summary
print("Random Forest dashboard flow visualization saved.")