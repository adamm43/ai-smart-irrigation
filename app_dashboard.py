import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── Path setup (works regardless of how streamlit is launched) ────────────────
ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_preparation import (
    load_and_clean_data, get_raw_dataframe, detect_target_column
)
from src.regression_model import (
    train_linear_regression, train_ridge,
    train_random_forest, tune_random_forest,
    train_gradient_boosting, tune_gradient_boosting,
    evaluate_regression, cross_validate_model,
    get_feature_importance, save_model,
)
from src.neural_network import build_nn, train_nn, evaluate_nn
from src.evaluation import compute_metrics, build_leaderboard, residual_analysis
from src.visualizations import (
    plot_distribution, plot_correlation_heatmap, plot_target_vs_feature,
    plot_missing_values, plot_boxplots, plot_target_by_category,
    plot_actual_vs_predicted, plot_residuals, plot_feature_importance,
    plot_model_comparison, plot_nn_history, plot_prediction_vs_index,
    plot_irrigation_rules,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="💧 Smart Irrigation AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title{font-size:2.2rem;font-weight:800;color:#1565C0;margin-bottom:0;}
.sub-title{font-size:1rem;color:#546E7A;margin-top:0;}
.metric-box{background:#F0F4FF;border-radius:10px;padding:14px 18px;
            border-left:5px solid #1565C0;margin:4px 0;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    DATA_PATH = st.text_input(
        "Dataset path",
        value="data/raw/smart_irrigation_4000.csv",
    )
    st.markdown("---")
    st.markdown("### 🤖 Models to train")
    run_lr    = st.checkbox("Linear Regression",  value=True)
    run_ridge = st.checkbox("Ridge Regression",   value=True)
    run_rf    = st.checkbox("Random Forest",      value=True)
    run_gbm   = st.checkbox("Gradient Boosting",  value=True)
    run_nn    = st.checkbox("Neural Network",     value=True)

    st.markdown("---")
    st.markdown("### 🎛️ Tuning")
    use_tuning = st.checkbox("Hyperparameter tuning (slower)", value=False)

    st.markdown("---")
    st.markdown("### 🧠 Neural Network")
    nn_epochs  = st.slider("Max Epochs",      50, 500, 150, 25)
    nn_layer1  = st.slider("Layer 1 neurons", 32, 256, 128, 16)
    nn_layer2  = st.slider("Layer 2 neurons", 16, 128,  64,  8)
    nn_layer3  = st.slider("Layer 3 neurons",  8,  64,  32,  8)
    nn_dropout = st.slider("Dropout rate",   0.0,  0.6, 0.3, 0.05)
    nn_batch   = st.slider("Batch size",       8,  128,  32,  8)

    st.markdown("---")
    cv_folds = st.slider("CV folds", 3, 10, 5)
    run_pipeline = st.button("🚀 Train All Models", type="primary",
                              use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">💧 Smart Irrigation AI Dashboard</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Prédiction du volume d\'eau nécessaire pour l\'irrigation '
    '— Machine Learning & Deep Learning</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Dataset not found: `{DATA_PATH}`")
    st.info("👉 Place `smart_irrigation_4000.csv` in `data/raw/`")
    st.stop()

try:
    with st.spinner("Chargement et prétraitement des données…"):
        X_train, X_test, y_train, y_test, preprocessor, feature_names = \
            load_and_clean_data(DATA_PATH)
        df_raw     = get_raw_dataframe(DATA_PATH)
        target_col = detect_target_column(df_raw)
    st.success(f"✅ Dataset chargé — {len(df_raw):,} lignes | Target: **{target_col}**")
except Exception as e:
    st.error(f"❌ Erreur: {e}")
    st.exception(e)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for key, val in [
    ("trained_models", {}), ("metrics_list", []),
    ("nn_history", None),   ("feature_imp_df", None),
    ("cv_results", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
if run_pipeline:
    for key in ["trained_models", "metrics_list", "nn_history",
                "feature_imp_df", "cv_results"]:
        st.session_state[key] = {} if key in ("trained_models", "cv_results") \
                                 else ([] if key == "metrics_list" else None)

    total        = sum([run_lr, run_ridge, run_rf, run_gbm, run_nn])
    done_counter = [0]
    progress     = st.progress(0)
    status       = st.status("Entraînement des modèles…", expanded=True)

    def _register(name, model, y_pred):
        st.session_state.metrics_list.append(
            compute_metrics(y_test, y_pred, model_name=name)
        )
        st.session_state.trained_models[name] = model
        done_counter[0] += 1
        progress.progress(done_counter[0] / total)

    if run_lr:
        status.write("Training Linear Regression…")
        m  = train_linear_regression(X_train, y_train)
        cv = cross_validate_model(m, X_train, y_train, cv=cv_folds)
        st.session_state.cv_results["Linear Regression"] = cv
        _register("Linear Regression", m, m.predict(X_test))

    if run_ridge:
        status.write("Training Ridge Regression…")
        m  = train_ridge(X_train, y_train)
        cv = cross_validate_model(m, X_train, y_train, cv=cv_folds)
        st.session_state.cv_results["Ridge Regression"] = cv
        _register("Ridge Regression", m, m.predict(X_test))

    if run_rf:
        if use_tuning:
            status.write("Tuning Random Forest…")
            m = tune_random_forest(X_train, y_train, cv=cv_folds, n_iter=15)
        else:
            status.write("Training Random Forest…")
            m = train_random_forest(X_train, y_train)
        cv  = cross_validate_model(m, X_train, y_train, cv=cv_folds)
        st.session_state.cv_results["Random Forest"] = cv
        imp = get_feature_importance(m, feature_names)
        if st.session_state.feature_imp_df is None:
            st.session_state.feature_imp_df = imp
        _register("Random Forest", m, m.predict(X_test))

    if run_gbm:
        if use_tuning:
            status.write("Tuning Gradient Boosting…")
            m = tune_gradient_boosting(X_train, y_train, cv=cv_folds, n_iter=15)
        else:
            status.write("Training Gradient Boosting…")
            m = train_gradient_boosting(X_train, y_train)
        cv  = cross_validate_model(m, X_train, y_train, cv=cv_folds)
        st.session_state.cv_results["Gradient Boosting"] = cv
        imp = get_feature_importance(m, feature_names)
        st.session_state.feature_imp_df = imp
        _register("Gradient Boosting", m, m.predict(X_test))

    if run_nn:
        status.write("Building and training Neural Network…")
        nn = build_nn(
            input_dim=X_train.shape[1],
            layer1=nn_layer1, layer2=nn_layer2, layer3=nn_layer3,
            dropout_rate=nn_dropout,
        )
        nn, history = train_nn(
            nn, X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=nn_epochs, batch_size=nn_batch,
        )
        st.session_state.nn_history = history
        nn_m = evaluate_nn(nn, X_test, y_test)
        st.session_state.metrics_list.append(
            compute_metrics(y_test, nn_m["y_pred"], "Neural Network")
        )
        st.session_state.trained_models["Neural Network"] = nn
        done_counter[0] += 1
        progress.progress(done_counter[0] / total)

    status.update(label="✅ Entraînement terminé!", state="complete")
    progress.empty()
    st.success(f"🎉 {total} modèle(s) entraîné(s) avec succès!")

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset",
    "🔍 Analyse EDA",
    "🏆 Comparaison Modèles",
    "🧠 Réseau de Neurones",
    "🎯 Prédiction Live",
    "📈 Importance Features",
])

# ── TAB 1 — DATASET ──────────────────────────────────────────────────────────
with tab1:
    st.header("📊 Vue d'ensemble du Dataset")
    st.markdown("""
    **Dataset:** `smart_irrigation_4000.csv`  
    **Objectif:** Prédire le volume d'eau (litres/jour) nécessaire pour irriguer un champ,  
    en fonction des conditions climatiques, du type de culture et du sol.
    """)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lignes",        f"{df_raw.shape[0]:,}")
    c2.metric("Features",      df_raw.shape[1] - 1)
    c3.metric("Target",        target_col)
    c4.metric("Train",         f"{len(X_train):,}")
    c5.metric("Test",          f"{len(X_test):,}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Aperçu des données")
        st.dataframe(df_raw.head(10), use_container_width=True)
    with col_r:
        st.subheader("Statistiques descriptives")
        st.dataframe(df_raw.describe().T.round(2), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Valeurs manquantes")
        st.pyplot(plot_missing_values(df_raw), use_container_width=True)
    with col_b:
        st.subheader(f"Distribution de la cible: {target_col}")
        st.pyplot(plot_distribution(df_raw, target_col), use_container_width=True)

    st.subheader("Types de données")
    st.dataframe(pd.DataFrame({
        "Colonne":   df_raw.dtypes.index,
        "Type":      df_raw.dtypes.values.astype(str),
        "Non-Null":  df_raw.notnull().sum().values,
        "Manquant":  df_raw.isnull().sum().values,
        "Unique":    df_raw.nunique().values,
    }), use_container_width=True)

    st.subheader("💧 Patterns d'irrigation")
    st.pyplot(plot_irrigation_rules(df_raw, target_col), use_container_width=True)


# ── TAB 2 — EDA ──────────────────────────────────────────────────────────────
with tab2:
    st.header("🔍 Analyse Exploratoire des Données (EDA)")

    numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df_raw.select_dtypes(include="object").columns.tolist()

    st.subheader("Distribution des features")
    sel_col = st.selectbox("Choisir une feature", numeric_cols, key="eda_dist")
    st.pyplot(plot_distribution(df_raw, sel_col), use_container_width=True)

    st.subheader("Détection des outliers — Boxplots")
    st.pyplot(plot_boxplots(df_raw), use_container_width=True)

    st.subheader("Matrice de corrélation")
    st.pyplot(plot_correlation_heatmap(df_raw), use_container_width=True)

    if target_col in numeric_cols:
        st.subheader("Feature vs Cible")
        other = [c for c in numeric_cols if c != target_col]
        sel_feat = st.selectbox("Choisir feature", other, key="eda_feat")
        st.pyplot(plot_target_vs_feature(df_raw, sel_feat, target_col),
                  use_container_width=True)

    if cat_cols:
        st.subheader("Cible par catégorie")
        sel_cat = st.selectbox("Choisir variable catégorielle", cat_cols, key="eda_cat")
        st.pyplot(plot_target_by_category(df_raw, sel_cat, target_col),
                  use_container_width=True)


# ── TAB 3 — MODEL COMPARISON ─────────────────────────────────────────────────
with tab3:
    st.header("🏆 Comparaison des Modèles")

    if not st.session_state.metrics_list:
        st.info("👈 Cliquez **Train All Models** dans la sidebar pour commencer.")
    else:
        lb = build_leaderboard(st.session_state.metrics_list)

        st.subheader("Leaderboard — Classement des modèles")
        def _highlight(s):
            best = (s == s.max()) if s.name == "R²" else (s == s.min())
            return ["background-color:#E8F5E9;font-weight:bold" if v else "" for v in best]
        styled = lb.style.apply(_highlight, subset=["MSE","RMSE","MAE","R²"]) \
                         .format({"MSE":"{:.2f}","RMSE":"{:.2f}",
                                  "MAE":"{:.2f}","R²":"{:.4f}","MAPE%":"{:.2f}%"})
        st.dataframe(styled, use_container_width=True, hide_index=True)

        best = lb.iloc[0]
        st.success(f"🏆 Meilleur modèle: **{best['Model']}** | R²: {best['R²']} | RMSE: {best['RMSE']} L")

        st.subheader("Comparaison visuelle")
        st.pyplot(plot_model_comparison(lb), use_container_width=True)

        if st.session_state.cv_results:
            st.subheader(f"Validation Croisée ({cv_folds}-Fold)")
            cv_rows = [{
                "Modèle": name,
                "CV RMSE": f"{cv['CV_RMSE_mean']:.2f} ± {cv['CV_RMSE_std']:.2f}",
                "CV MAE":  f"{cv['CV_MAE_mean']:.2f}  ± {cv['CV_MAE_std']:.2f}",
                "CV R²":   f"{cv['CV_R2_mean']:.4f} ± {cv['CV_R2_std']:.4f}",
            } for name, cv in st.session_state.cv_results.items()]
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

        st.subheader("Analyse détaillée par modèle")
        sel_model = st.selectbox("Choisir modèle", list(st.session_state.trained_models.keys()))
        m = st.session_state.trained_models[sel_model]
        y_pred = m.predict(X_test, verbose=0).flatten() \
                 if sel_model == "Neural Network" else m.predict(X_test)

        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(plot_actual_vs_predicted(y_test.values, y_pred, sel_model),
                      use_container_width=True)
        with col_b:
            st.pyplot(plot_residuals(y_test.values, y_pred, sel_model),
                      use_container_width=True)

        st.pyplot(plot_prediction_vs_index(y_test.values, y_pred, 100, sel_model),
                  use_container_width=True)

        with st.expander("📋 Table des résidus"):
            res = residual_analysis(y_test, y_pred)
            st.dataframe(res.head(50).style.format({
                "actual":"{:.1f}","predicted":"{:.1f}",
                "residual":"{:.1f}","abs_residual":"{:.1f}","pct_error":"{:.2f}%"
            }), use_container_width=True)


# ── TAB 4 — NEURAL NETWORK ───────────────────────────────────────────────────
with tab4:
    st.header("🧠 Réseau de Neurones — Détails")

    if st.session_state.nn_history is None:
        st.info("👈 Activez **Neural Network** et cliquez **Train All Models**.")
    else:
        h = st.session_state.nn_history.history
        n_epochs = len(h["loss"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Epochs",          n_epochs)
        c2.metric("Train loss final", f"{h['loss'][-1]:.4f}")
        if "val_loss" in h:
            c3.metric("Val loss final",  f"{h['val_loss'][-1]:.4f}")
            c4.metric("Best epoch",      int(np.argmin(h["val_loss"])) + 1)

        st.subheader("Courbes d'entraînement")
        st.pyplot(plot_nn_history(st.session_state.nn_history),
                  use_container_width=True)

        if "Neural Network" in st.session_state.trained_models:
            nn = st.session_state.trained_models["Neural Network"]
            st.subheader("Architecture du réseau")
            rows = []
            for l in nn.layers:
                try:
                    shape = str(l.output_shape)
                except Exception:
                    shape = "N/A"
                rows.append({
                    "Couche":        l.name,
                    "Type":          l.__class__.__name__,
                    "Output Shape":  shape,
                    "Paramètres":    f"{l.count_params():,}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.metric("Total paramètres", f"{nn.count_params():,}")

        with st.expander("📋 Historique complet"):
            st.dataframe(pd.DataFrame(h), use_container_width=True)


# ── TAB 5 — LIVE PREDICTION ──────────────────────────────────────────────────
with tab5:
    st.header("🎯 Prédiction Live — Volume d'eau nécessaire")
    st.markdown("Entrez les conditions de votre champ et obtenez une prédiction instantanée.")

    if not st.session_state.trained_models:
        st.info("👈 Entraînez les modèles d'abord.")
    else:
        model_choice = st.selectbox(
            "Modèle de prédiction",
            list(st.session_state.trained_models.keys()),
            key="pred_model",
        )

        st.subheader("📝 Paramètres du champ")
        user_inputs = {}
        n_cols = 3
        chunks = [feature_names[i:i+n_cols] for i in range(0, len(feature_names), n_cols)]
        for chunk in chunks:
            cols = st.columns(len(chunk))
            for col, feat in zip(cols, chunk):
                default = float(X_train[feat].median())
                user_inputs[feat] = col.number_input(
                    feat, value=default, format="%.4f", key=f"inp_{feat}"
                )

        if st.button("💧 Prédire le volume d'eau", type="primary"):
            input_df = pd.DataFrame([user_inputs])
            model    = st.session_state.trained_models[model_choice]
            pred     = model.predict(input_df, verbose=0).flatten()[0] \
                       if model_choice == "Neural Network" \
                       else model.predict(input_df)[0]
            pred     = max(0, pred)

            st.success(f"### 💧 Volume prédit: **{pred:.1f} litres/jour**")

            # Interpretation
            if pred < 200:
                st.info("🟢 Faible besoin en eau — conditions favorables")
            elif pred < 800:
                st.warning("🟡 Besoin modéré — irrigation recommandée")
            else:
                st.error("🔴 Besoin élevé — irrigation urgente nécessaire")

            pct = float(np.mean(y_train < pred) * 100)
            st.info(
                f"Ce champ nécessite plus d'eau que **{pct:.1f}%** des champs "
                f"(min: {y_train.min():.0f}L — max: {y_train.max():.0f}L — "
                f"moyenne: {y_train.mean():.0f}L)"
            )

        with st.expander("🔍 Prédictions sur 20 échantillons test"):
            model = st.session_state.trained_models[model_choice]
            preds = model.predict(X_test[:20], verbose=0).flatten() \
                    if model_choice == "Neural Network" \
                    else model.predict(X_test[:20])
            sdf = pd.DataFrame({
                "Actual (L)":    y_test[:20].values.round(1),
                "Predicted (L)": preds.round(1),
                "Error (L)":     np.abs(y_test[:20].values - preds).round(1),
            })
            st.dataframe(sdf.style.format("{:.1f}"), use_container_width=True)


# ── TAB 6 — FEATURE IMPORTANCE ───────────────────────────────────────────────
with tab6:
    st.header("📈 Importance des Features")

    if st.session_state.feature_imp_df is None:
        st.info("👈 Entraînez Random Forest ou Gradient Boosting.")
    else:
        imp_df = st.session_state.feature_imp_df
        top_n  = st.slider("Top N features", 5, len(imp_df), min(15, len(imp_df)))
        st.pyplot(plot_feature_importance(imp_df, top_n=top_n),
                  use_container_width=True)

        st.subheader("Table complète")
        st.dataframe(
            imp_df.style.format({"importance": "{:.6f}"}),
            use_container_width=True,
        )

        st.markdown("""
        **Interprétation:**
        - `field_area_ha` → Plus le champ est grand, plus il faut d'eau *(logique physique)*
        - `soil_moisture_%` → Plus le sol est humide, moins on irrigue *(logique agronomique)*  
        - `rainfall_mm` → La pluie réduit le besoin d'irrigation *(logique climatique)*
        - `temperature_C` → Chaleur = évaporation = plus d'eau *(logique climatique)*
        - `crop_type` → Riz > Maïs > Blé en besoin hydrique *(agronomie)*
        """)

        if "Linear Regression" in st.session_state.trained_models:
            st.subheader("Coefficients — Régression Linéaire")
            lr = st.session_state.trained_models["Linear Regression"]
            coef_df = pd.DataFrame({
                "Feature":     feature_names,
                "Coefficient": lr.coef_,
            }).sort_values("Coefficient", key=abs, ascending=False)
            fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names)*0.4)))
            colors = ["#2196F3" if c >= 0 else "#FF5722" for c in coef_df["Coefficient"]]
            ax.barh(coef_df["Feature"][::-1], coef_df["Coefficient"][::-1],
                    color=colors[::-1])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Coefficients — Régression Linéaire",
                         fontsize=13, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>💧 Smart Irrigation AI · Optimisation de l'irrigation intelligente · "
    "scikit-learn + TensorFlow + Streamlit</small></center>",
    unsafe_allow_html=True,
)