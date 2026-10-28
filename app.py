import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import database


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Digital Twin of the Human Brain",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Digital Twin of the Human Brain")

st.subheader("Brain-Inspired Neural Network & AI Simulation")

st.info(
    "Educational simulation using artificial neurons and "
    "synthetic data. It is not a complete replica of a real brain."
)


# =====================================================
# SIDEBAR CONTROLS
# =====================================================

st.sidebar.title("⚙️ Simulation Controls")

neuron_count = st.sidebar.slider(
    "Number of artificial neurons",
    10, 80, 30, 5
)

connection_density = st.sidebar.slider(
    "Connection density",
    0.05, 0.40, 0.15, 0.05
)

input_strength = st.sidebar.slider(
    "Input strength",
    0.0, 1.0, 0.6, 0.1
)

noise_level = st.sidebar.slider(
    "Activity noise",
    0.0, 0.5, 0.1, 0.05
)

simulation_steps = st.sidebar.slider(
    "Simulation time steps",
    10, 100, 40, 10
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=99999,
    value=42
)

if st.sidebar.button("🔄 Generate New Simulation"):
    st.session_state.pop("simulation_data", None)


# =====================================================
# SIMULATION ENGINE
# =====================================================

@st.cache_data
def generate_simulation(
    n, density, strength, noise, steps, seed
):
    rng = np.random.default_rng(seed)

    graph = nx.erdos_renyi_graph(
        n=n,
        p=density,
        seed=seed
    )

    positions = rng.normal(
        0, 1, size=(n, 3)
    )

    positions[:, 0] *= 1.3
    positions[:, 1] *= 0.8
    positions[:, 2] *= 0.9

    adjacency = nx.to_numpy_array(graph)

    activity = np.zeros((steps, n))
    activity[0] = rng.random(n) * strength

    for t in range(1, steps):
        previous = activity[t - 1]

        if adjacency.sum() > 0:
            neighbors = (
                adjacency @ previous
            ) / np.maximum(
                adjacency.sum(axis=1), 1
            )
        else:
            neighbors = np.zeros(n)

        updated = (
            0.65 * previous
            + 0.25 * neighbors
            + 0.10 * strength
            + rng.normal(0, noise, n)
        )

        activity[t] = np.clip(updated, 0, 1)

    return graph, positions, activity


settings = (
    neuron_count,
    connection_density,
    input_strength,
    noise_level,
    simulation_steps,
    int(random_seed)
)

if (
    "simulation_data" not in st.session_state
    or st.session_state.get("settings") != settings
):
    st.session_state.simulation_data = generate_simulation(
        *settings
    )
    st.session_state.settings = settings

graph, positions, activity = st.session_state.simulation_data


# =====================================================
# SUMMARY METRICS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("🧠 Artificial Neurons", graph.number_of_nodes())
col2.metric("🔗 Connections", graph.number_of_edges())
col3.metric("⚡ Average Activity", f"{activity[-1].mean():.2f}")
col4.metric(
    "✨ Active Neurons",
    int(np.sum(activity[-1] > 0.5))
)


# =====================================================
# 3D NEURAL NETWORK
# =====================================================

st.header("🌐 3D Artificial Neural Network")

edge_x, edge_y, edge_z = [], [], []

for source, target in graph.edges():
    edge_x.extend([
        positions[source, 0],
        positions[target, 0],
        None
    ])
    edge_y.extend([
        positions[source, 1],
        positions[target, 1],
        None
    ])
    edge_z.extend([
        positions[source, 2],
        positions[target, 2],
        None
    ])

edges = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode="lines",
    line=dict(color="rgba(120,160,220,0.4)", width=2),
    hoverinfo="none",
    name="Connections"
)

nodes = go.Scatter3d(
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
    mode="markers",
    marker=dict(
        size=6,
        color=activity[-1],
        colorscale="Turbo",
        cmin=0,
        cmax=1,
        colorbar=dict(title="Activity")
    ),
    text=[
        f"Neuron {i + 1}: {activity[-1, i]:.3f}"
        for i in range(neuron_count)
    ],
    hovertemplate="%{text}<extra></extra>",
    name="Artificial Neurons"
)

fig = go.Figure(data=[edges, nodes])

fig.update_layout(
    height=550,
    margin=dict(l=0, r=0, t=20, b=0),
    paper_bgcolor="#0b1020",
    font=dict(color="white"),
    scene=dict(aspectmode="data")
)

st.plotly_chart(fig, use_container_width=True)


# =====================================================
# ACTIVITY GRAPH
# =====================================================

st.header("⚡ Neural Activity Over Time")

mean_activity = activity.mean(axis=1)

activity_fig = go.Figure()

activity_fig.add_trace(
    go.Scatter(
        x=np.arange(1, len(mean_activity) + 1),
        y=mean_activity,
        mode="lines",
        name="Average Activity",
        line=dict(width=3)
    )
)

activity_fig.update_layout(
    xaxis_title="Time Step",
    yaxis_title="Activity (0 to 1)",
    yaxis=dict(range=[0, 1]),
    height=350,
    paper_bgcolor="#0b1020",
    plot_bgcolor="#11182b",
    font=dict(color="white")
)

st.plotly_chart(activity_fig, use_container_width=True)


# =====================================================
# ACTIVITY HEATMAP
# =====================================================

st.header("🔥 Neural Activity Heatmap")

heatmap = px.imshow(
    activity.T,
    aspect="auto",
    origin="lower",
    color_continuous_scale="Turbo",
    labels={
        "x": "Time Step",
        "y": "Artificial Neuron",
        "color": "Activity"
    }
)

heatmap.update_layout(height=400)

st.plotly_chart(heatmap, use_container_width=True)


# =====================================================
# SAVE EXPERIMENT TO SQLITE
# =====================================================

st.header("💾 Save Experiment")

st.write(
    "Save the current simulation settings and all generated "
    "neural activity records to the SQLite database."
)

if st.button("💾 Save Current Simulation"):

    simulation_id = database.save_simulation(
        neuron_count=neuron_count,
        connection_density=connection_density,
        input_strength=input_strength,
        noise_level=noise_level,
        simulation_steps=simulation_steps,
        random_seed=int(random_seed),
        activity=activity
    )

    st.success(
        f"Experiment saved successfully! "
        f"Experiment ID: {simulation_id}"
    )


# =====================================================
# AI TRAINING
# =====================================================

st.header("🤖 AI Learning Demonstration")

st.write(
    "Train a small artificial neural network to classify "
    "synthetic data. This is separate from the simulated neural network above."
)

sample_count = st.select_slider(
    "Synthetic dataset size",
    options=[200, 400, 600, 800, 1000],
    value=600
)

if st.button("🚀 Train AI Model"):

    with st.spinner("Training AI model..."):

        X, y = make_classification(
            n_samples=sample_count,
            n_features=8,
            n_informative=5,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

        model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        database.save_ai_result(
            dataset_size=sample_count,
            accuracy=float(accuracy)
        )

        st.session_state.ai_accuracy = accuracy

    st.success("AI model trained and its result saved!")

if "ai_accuracy" in st.session_state:
    st.metric(
        "AI Test Accuracy",
        f"{st.session_state.ai_accuracy * 100:.2f}%"
    )


# =====================================================
# EXPERIMENT HISTORY
# =====================================================

st.header("📚 Saved Experiment History")

experiments = database.get_simulations()

if experiments:

    history_df = pd.DataFrame(experiments)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    selected_id = st.selectbox(
        "Choose an experiment to inspect",
        options=[row["id"] for row in experiments]
    )

    if st.button("🔍 Load Selected Experiment Activity"):

        records = database.get_simulation_activity(
            int(selected_id)
        )

        if records:
            records_df = pd.DataFrame(records)

            st.dataframe(
                records_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Saved Activity",
                data=records_df.to_csv(index=False).encode("utf-8"),
                file_name=f"simulation_{selected_id}_activity.csv",
                mime="text/csv"
            )
        else:
            st.warning("No activity records found for this experiment.")

else:
    st.info(
        "No experiments saved yet. Click 'Save Current Simulation' "
        "to store your first experiment."
    )


# =====================================================
# AI TRAINING HISTORY
# =====================================================

st.header("📈 AI Training History")

ai_results = database.get_ai_results()

if ai_results:

    ai_df = pd.DataFrame(ai_results)

    st.dataframe(
        ai_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download AI Training History",
        data=ai_df.to_csv(index=False).encode("utf-8"),
        file_name="ai_training_history.csv",
        mime="text/csv"
    )

else:
    st.info(
        "No AI training results saved yet. "
        "Train the model above to record your first result."
    )


# =====================================================
# EXPORT CURRENT ACTIVITY
# =====================================================

st.header("📥 Export Current Simulation")

current_df = pd.DataFrame(
    activity,
    columns=[
        f"Neuron {i + 1}"
        for i in range(neuron_count)
    ]
)

current_df.insert(
    0,
    "Time Step",
    np.arange(1, simulation_steps + 1)
)

st.download_button(
    "⬇️ Download Current Activity as CSV",
    data=current_df.to_csv(index=False).encode("utf-8"),
    file_name="brain_simulation_activity.csv",
    mime="text/csv"
)


# =====================================================
# PROJECT INFORMATION
# =====================================================

with st.expander("ℹ️ About This Project"):

    st.write("""
    **Project:** Digital Twin of the Human Brain

    **Database:** SQLite

    **Database file:** brain_database.db

    **Tables:**
    - simulations
    - neural_activity
    - ai_training_results

    **Technologies:** Python, Streamlit, SQLite, NumPy,
    NetworkX, Plotly, Pandas and Scikit-learn.

    This project is an educational brain-inspired simulation.
    It does not replicate the full biology or functionality
    of a real human brain, and it does not use real brain data.
    """)

st.divider()

st.caption(
    "Digital Twin of the Human Brain | "
    "Educational AI Simulation with SQLite"
)