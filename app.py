import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config for a premium look
st.set_page_config(
    page_title="GA Feature & Hyperparameter Optimizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom premium CSS for styling
st.markdown("""
    <style>
        /* Custom font and clean styling */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #888888;
            margin-bottom: 2rem;
        }
        
        /* Metric card styling */
        .metric-card {
            background-color: #1E1E24;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #2E2E38;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease-in-out;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        /* Highlight sections */
        .highlight-text {
            color: #FF4B4B;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Define Hyperparameter Spaces for 9 models
PARAM_SPACE = {
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan'],
    },
    'SVM': {
        'C':      [0.1, 0.5, 1.0, 5.0, 10.0],
        'gamma':  ['scale', 'auto'],
        'kernel': ['rbf', 'linear', 'poly'],
    },
    'RF': {
        'n_estimators':      [50, 100, 200],
        'max_depth':         [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
    },
    'LR': {
        'C':        [0.01, 0.1, 1.0, 10.0],
        'penalty':  ['l2'],
        'solver':   ['lbfgs', 'liblinear'],
    },
    'DT': {
        'criterion':         ['gini', 'entropy'],
        'max_depth':         [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
    },
    'Ada': {
        'n_estimators':  [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
    },
    'GB': {
        'n_estimators':  [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth':     [3, 5],
    },
    'NB': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation':         ['tanh', 'relu'],
        'learning_rate_init': [0.001, 0.01],
    },
}

def decode_params(model_name, param_genes):
    """Integer gene array → concrete hyperparameter dict."""
    params = {}
    for gene, (pname, options) in zip(param_genes, PARAM_SPACE[model_name].items()):
        params[pname] = options[int(gene) % len(options)]
    return params

def build_model(model_name, param_genes):
    """Instantiate a model from its name + param genes."""
    p = decode_params(model_name, param_genes)
    if model_name == 'KNN':
        return KNeighborsClassifier(**p)
    elif model_name == 'SVM':
        return SVC(random_state=42, max_iter=2000, **p)
    elif model_name == 'RF':
        return RandomForestClassifier(n_jobs=-1, random_state=42, **p)
    elif model_name == 'LR':
        return LogisticRegression(random_state=42, max_iter=1000, **p)
    elif model_name == 'DT':
        return DecisionTreeClassifier(random_state=42, **p)
    elif model_name == 'Ada':
        return AdaBoostClassifier(random_state=42, **p)
    elif model_name == 'GB':
        return GradientBoostingClassifier(random_state=42, **p)
    elif model_name == 'NB':
        return GaussianNB(**p)
    elif model_name == 'MLP':
        return MLPClassifier(random_state=42, max_iter=500, early_stopping=True, **p)

def default_model(model_name):
    """Return a model with sklearn defaults for baseline comparison."""
    if model_name == 'KNN':
        return KNeighborsClassifier()
    elif model_name == 'SVM':
        return SVC(random_state=42, max_iter=2000)
    elif model_name == 'RF':
        return RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=42)
    elif model_name == 'LR':
        return LogisticRegression(random_state=42, max_iter=1000)
    elif model_name == 'DT':
        return DecisionTreeClassifier(random_state=42)
    elif model_name == 'Ada':
        return AdaBoostClassifier(random_state=42)
    elif model_name == 'GB':
        return GradientBoostingClassifier(random_state=42)
    elif model_name == 'NB':
        return GaussianNB()
    elif model_name == 'MLP':
        return MLPClassifier(random_state=42, max_iter=500, early_stopping=True)

# GA chromosome helpers
def random_chromosome(n_features, model_name):
    feat_bits   = np.random.randint(0, 2, n_features)
    param_genes = np.array([
        np.random.randint(0, len(opts))
        for opts in PARAM_SPACE[model_name].values()
    ])
    return np.concatenate([feat_bits, param_genes])

def split_chromosome(chromo, n_features):
    return chromo[:n_features].astype(int), chromo[n_features:].astype(int)

def fitness(chromo, model_name, X, y, n_features, cache=None):
    """Joint fitness with caching and parallel execution: 0.99 * CV_accuracy + 0.01 * feature_reduction_bonus"""
    if cache is not None:
        key = (model_name, tuple(chromo.tolist()))
        if key in cache:
            return cache[key]

    feat_bits, param_genes = split_chromosome(chromo, n_features)
    if not any(feat_bits):
        val = 0.0
    else:
        model  = build_model(model_name, param_genes)
        try:
            # Parallelize cross validation folds using n_jobs=-1
            score  = cross_val_score(model, X[:, feat_bits == 1], y, cv=5, n_jobs=-1).mean()
        except Exception:
            score = 0.0
        bonus  = 1.0 - feat_bits.sum() / n_features
        val = 0.99 * score + 0.01 * bonus

    if cache is not None:
        cache[key] = val
    return val

def crossover(p1, p2):
    n = len(p1)
    pt1, pt2 = sorted(np.random.choice(range(1, n), 2, replace=False))
    c1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]])
    c2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]])
    return c1, c2

def mutate(chromo, n_features, model_name, rate=0.05):
    chromo = chromo.copy().astype(int)
    for i in range(n_features):
        if np.random.rand() < rate:
            chromo[i] ^= 1
    for j, opts in enumerate(PARAM_SPACE[model_name].values()):
        if np.random.rand() < rate:
            chromo[n_features + j] = np.random.randint(0, len(opts))
    return chromo

# Caching dataset loaders
@st.cache_data
def get_breast_cancer():
    bc = load_breast_cancer()
    return StandardScaler().fit_transform(bc.data), bc.target, bc.feature_names.tolist()

@st.cache_data
def get_ionosphere():
    # Fast load directly from UCI database
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    df = pd.read_csv(url, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df.iloc[:, -1])
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    return StandardScaler().fit_transform(X), y, feature_names

def parse_uploaded_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        # Drop standard index-like columns
        drop_cols = ['time', 'id', 'unnamed: 0', 'index']
        df = df.drop(columns=[c for c in df.columns if c.lower() in drop_cols], errors='ignore')
        
        # Try to guess label
        label_hints = ['class', 'label', 'target', 'outcome', 'diagnosis', 'category', 'fraud', 'y']
        target_col = None
        for hint in label_hints:
            matches = [c for c in df.columns if hint in c.lower()]
            if matches:
                target_col = matches[0]
                break
        
        if not target_col:
            target_col = df.columns[-1]
            
        y_raw = df[target_col]
        df_feats = df.drop(columns=[target_col])
        
        # Encode label
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        
        # Numeric conversions for features
        df_feats = df_feats.apply(pd.to_numeric, errors='coerce')
        df_feats = df_feats.dropna(axis=1, thresh=int(0.8 * len(df_feats)))
        df_feats = df_feats.fillna(df_feats.median(numeric_only=True))
        
        X = StandardScaler().fit_transform(df_feats.values.astype(float))
        return X, y, df_feats.columns.tolist()
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        return None

# App UI Header
st.markdown('<div class="main-title">🧬 Genetic Algorithm Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Joint Feature Selection & Hyperparameter Optimization Dashboard</div>', unsafe_allow_html=True)

# Sidebar layout
st.sidebar.markdown("<h1 style='text-align: center; font-size: 4rem; margin: 0px; padding: 0px;'>🧬</h1>", unsafe_allow_html=True)
st.sidebar.header("Configuration")

# Dataset Selection
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ["Breast Cancer (Built-in)", "Ionosphere (OpenML)", "Upload Custom CSV"]
)

X, y, feature_names = None, None, []
if dataset_choice == "Breast Cancer (Built-in)":
    X, y, feature_names = get_breast_cancer()
    st.info("Loaded Breast Cancer Dataset (569 samples, 30 features)")
elif dataset_choice == "Ionosphere (OpenML)":
    with st.spinner("Loading Ionosphere Dataset..."):
        X, y, feature_names = get_ionosphere()
    st.info("Loaded Ionosphere Dataset (351 samples, 34 features)")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        parsed = parse_uploaded_file(uploaded_file)
        if parsed:
            X, y, feature_names = parsed
            st.success(f"Successfully loaded {uploaded_file.name} ({X.shape[0]} samples, {X.shape[1]} features)")
    else:
        st.warning("Please upload a CSV file to proceed.")

# Model Selection - Now supports all 9 models!
selected_models = st.sidebar.multiselect(
    "Select Classifiers",
    ["KNN", "SVM", "RF", "LR", "DT", "Ada", "GB", "NB", "MLP"],
    default=["KNN", "SVM", "RF"]
)

# GA Parameters
st.sidebar.subheader("Genetic Algorithm Parameters")
pop_size = st.sidebar.slider("Population Size", min_value=10, max_value=100, value=20, step=5)
generations = st.sidebar.slider("Generations", min_value=5, max_value=50, value=10, step=5)
mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

# Execution Trigger
run_opt = st.sidebar.button("🚀 Run GA Optimization", use_container_width=True)

# Main Area
if X is not None and y is not None:
    # Display Dataset Metrics in a beautiful custom HTML structure (no truncation or leakage)
    classes, counts = np.unique(y, return_counts=True)
    class_dist = {int(c): int(count) for c, count in zip(classes, counts)}
    dist_str = ", ".join([f"<span class='highlight-text'>{k}</span>: {v}" for k, v in class_dist.items()])
    
    st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap;">
            <div class="metric-card" style="flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 0.9rem; color: #888888; text-transform: uppercase; letter-spacing: 1px;">Total Samples</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin-top: 0.5rem;">{X.shape[0]}</div>
            </div>
            <div class="metric-card" style="flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 0.9rem; color: #888888; text-transform: uppercase; letter-spacing: 1px;">Total Features</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin-top: 0.5rem;">{X.shape[1]}</div>
            </div>
            <div class="metric-card" style="flex: 1.5; min-width: 280px; text-align: center;">
                <div style="font-size: 0.9rem; color: #888888; text-transform: uppercase; letter-spacing: 1px;">Class Distribution</div>
                <div style="font-size: 1.4rem; font-weight: 600; color: #ffffff; margin-top: 0.9rem;">{dist_str}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if run_opt:
        if not selected_models:
            st.warning("Please select at least one classifier model to run.")
        else:
            results = []
            ga_histories = {}
            n_feat = X.shape[1]

            st.write("---")
            st.subheader("🧬 Run Progress")
            
            # Global evaluation cache to speed up GA optimization runs
            eval_cache = {}
            
            for m_name in selected_models:
                st.markdown(f"#### Running Optimization for <span class='highlight-text'>{m_name}</span>", unsafe_allow_html=True)
                
                # Baseline - optimized parallel runs
                baseline_acc = cross_val_score(default_model(m_name), X, y, cv=5, n_jobs=-1).mean()
                
                # GA Process Setup
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                pop = [random_chromosome(n_feat, m_name) for _ in range(pop_size)]
                best_chromo, best_fit = None, -1.0
                history = []
                
                for gen in range(generations):
                    # Cache evaluation values to skip re-running duplicate chromosomes
                    scores = [fitness(c, m_name, X, y, n_feat, cache=eval_cache) for c in pop]
                    order  = np.argsort(scores)[::-1]
                    pop    = [pop[i] for i in order]
                    gen_best = scores[order[0]]
                    history.append(gen_best)
                    
                    if gen_best > best_fit:
                        best_fit    = gen_best
                        best_chromo = pop[0].copy()
                        
                    # Re-populate
                    next_pop = pop[:2]
                    while len(next_pop) < pop_size:
                        i1, i2 = np.random.choice(max(2, pop_size // 2), 2, replace=False)
                        c1, c2 = crossover(pop[i1], pop[i2])
                        next_pop += [mutate(c1, n_feat, m_name, mutation_rate),
                                     mutate(c2, n_feat, m_name, mutation_rate)]
                    pop = next_pop[:pop_size]
                    
                    feat_bits, param_genes = split_chromosome(pop[0], n_feat)
                    params = decode_params(m_name, param_genes)
                    
                    progress_percentage = (gen + 1) / generations
                    progress_bar.progress(progress_percentage)
                    status_text.text(f"Generation {gen+1}/{generations} | Best Fitness: {gen_best:.4f} | Features: {int(feat_bits.sum())}/{n_feat}")
                
                feat_bits, param_genes = split_chromosome(best_chromo, n_feat)
                best_params = decode_params(m_name, param_genes)
                n_sel = int(feat_bits.sum())
                
                ga_acc = (cross_val_score(build_model(m_name, param_genes), X[:, feat_bits == 1], y, cv=5, n_jobs=-1).mean()
                          if n_sel > 0 else baseline_acc)
                
                delta = ga_acc - baseline_acc
                ga_histories[m_name] = history
                
                results.append({
                    "Model": m_name,
                    "Baseline Accuracy": baseline_acc,
                    "GA Optimized Accuracy": ga_acc,
                    "Delta": delta,
                    "Selected Features": n_sel,
                    "Best Hyperparameters": str(best_params),
                    "feature_mask": feat_bits
                })
                
                st.success(f"Finished {m_name} Optimization! Accuracy: {baseline_acc:.4f} ➡️ {ga_acc:.4f} (Delta: {delta:+.4f})")

            # Store results in a DataFrame
            df_results = pd.DataFrame(results)
            
            st.write("---")
            st.subheader("📊 Optimization Summary")
            
            # Interactive Plotly chart for accuracy comparison
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                x=df_results["Model"],
                y=df_results["Baseline Accuracy"],
                name="Baseline (All Features, Defaults)",
                marker_color="#4C72B0"
            ))
            fig_acc.add_trace(go.Bar(
                x=df_results["Model"],
                y=df_results["GA Optimized Accuracy"],
                name="GA Optimized (Selected Features, Tuned Params)",
                marker_color="#DD8452"
            ))
            fig_acc.update_layout(
                title="Model Accuracy Comparison",
                barmode="group",
                yaxis_title="5-Fold CV Accuracy",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            # Interactive Plotly chart for fitness curves
            st.subheader("📈 Fitness Progress over Generations")
            fig_fit = go.Figure()
            for m_name, history in ga_histories.items():
                fig_fit.add_trace(go.Scatter(
                    y=history,
                    mode='lines+markers',
                    name=m_name,
                    line=dict(width=3)
                ))
            fig_fit.update_layout(
                title="Best Fitness Tracker",
                xaxis_title="Generations",
                yaxis_title="Fitness Score",
                template="plotly_dark"
            )
            st.plotly_chart(fig_fit, use_container_width=True)

            # Show results table
            st.subheader("📋 Performance Table")
            st.dataframe(df_results[["Model", "Baseline Accuracy", "GA Optimized Accuracy", "Delta", "Selected Features", "Best Hyperparameters"]].style.format({
                "Baseline Accuracy": "{:.4f}",
                "GA Optimized Accuracy": "{:.4f}",
                "Delta": "{:+.4f}"
            }))
            
            # Show selected features for each model
            st.subheader("🔍 Selected Features Breakdown")
            for res in results:
                with st.expander(f"Selected Features for {res['Model']}"):
                    sel_indices = np.where(res['feature_mask'] == 1)[0]
                    sel_names = [feature_names[i] for i in sel_indices]
                    st.write(f"**Total Features Selected:** {len(sel_names)} / {len(feature_names)}")
                    st.write(", ".join(sel_names))
else:
    st.info("👈 Use the sidebar config to select a dataset and click 'Run GA Optimization'.")
