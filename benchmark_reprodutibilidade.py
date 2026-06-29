# Mede como o desempenho de modelos clássicos varia conforme a semente
# aleatória, treinando cada combinação (dataset, modelo, semente) e guardando
# acurácia e AUC. A ideia e rodar a MESMA configuração várias vezes mudando so
# a semente, pra depois medir o quanto o "ranking" entre os modelos balança.
#
# Detalhe que importa no Mac: cada modelo roda com n_jobs=1 e quem paraleliza
# é o laço de fora (joblib). Se os dois paralelizassem ao mesmo tempo, os
# núcleos brigariam entre si e ficaria mais lento, não mais rápido.

import os
import time
import warnings

import pandas as pd
from joblib import Parallel, delayed

import openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Setado para não mostrar os avisos pra não poluir a saáda
warnings.filterwarnings("ignore")

# IDs dos datasets na OpenML (suite CC18). São 10 bases tabulares de classificação.
DATASET_IDS = [31, 1464, 1494, 1510, 37, 44, 50, 151, 1461, 1590]

# 20 sementes por configuração. Quanto mais sementes, mais estavel fica
SEEDS = list(range(20))

N_CORES = -1                      # -1 = usa todos os nucleos no laço externo
RESULTS_PATH = "resultados.parquet"


def build_models(seed):
    # Cada modelo vem dentro de um pipeline com imputação, porque alguns
    # datasets da CC18 tem valores faltantes e os modelos não aceitam NaN.
    # A regressão ainda leva um StandardScaler (faz diferença pra ela; pras
    # árvores, não).
    return {
        "logreg": make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="mean"),
            RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=seed),
        ),
        "xgboost": make_pipeline(
            SimpleImputer(strategy="mean"),
            XGBClassifier(n_estimators=300, n_jobs=1, random_state=seed,
                          eval_metric="logloss", verbosity=0),
        ),
        "lightgbm": make_pipeline(
            SimpleImputer(strategy="mean"),
            LGBMClassifier(n_estimators=300, n_jobs=1, random_state=seed,
                           verbose=-1),
        ),
    }


def load_dataset(dataset_id):
    ds = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)

    # get_dummies resolve as colunas categoricas; dummy_na marca os faltantes
    # como categoria própria. factorize transforma o alvo em inteiros 0,1,2...
    X = pd.get_dummies(X, dummy_na=True)
    y = pd.factorize(y)[0]
    return X, y, ds.name


def run_one(dataset_id, model_name, seed):

    warnings.filterwarnings("ignore")

    X, y, ds_name = load_dataset(dataset_id)

    # A semente entra no split: ela decide quem vai pro treino e quem vai pro
    # teste. Como uso a mesma semente pra todos os modelos, eles enxergam a
    # MESMA partição - e isso que torna a comparação pareada depois.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    model = build_models(seed)[model_name]

    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)

    # AUC so faz sentido em problema binário; nos multiclasse o predict_proba
    # tem mais de 2 colunas e o roc_auc_score reclama - por isso o try/except.
    try:
        proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
    except Exception:
        auc = float("nan")

    return {
        "dataset_id": dataset_id,
        "dataset": ds_name,
        "model": model_name,
        "seed": seed,
        "accuracy": acc,
        "auc": auc,
        "train_seconds": round(elapsed, 3),
    }


def get_completed_tasks():
    # Se já existe um parquet de uma rodada anterior, leio quais combinacoes
    # ja foram feitas pra nao refazer. E o que permite rodar o experimento em
    # partes (fechar o Mac e continuar depois).
    if not os.path.exists(RESULTS_PATH):
        return set()
    df = pd.read_parquet(RESULTS_PATH)
    return set(zip(df.dataset_id, df.model, df.seed))


def main():
    done = get_completed_tasks()
    model_names = list(build_models(0).keys())

    # Baixo os datasets uma vez antes de paralelizar. Assim eles ficam no cache
    # local do openml e os processos filhos não tentam baixar o mesmo arquivo
    # ao mesmo tempo (o que dava erro de download concorrente).
    for d_id in DATASET_IDS:
        load_dataset(d_id)

    tasks = [
        (d, m, s)
        for d in DATASET_IDS
        for m in model_names
        for s in SEEDS
        if (d, m, s) not in done
    ]

    if not tasks:
        print("Todos os experimentos já foram concluídos.")
        return

    print(f"Executando {len(tasks)} tarefas ({len(done)} puladas).")

    # Paraleliza o laco externo: cada núcleo cuida de um treino inteiro.
    results = Parallel(n_jobs=N_CORES, verbose=10)(
        delayed(run_one)(d, m, s) for (d, m, s) in tasks
    )

    # Junto com o que já existia (se existia) e regravo o parquet inteiro.
    new_df = pd.DataFrame(results)
    if os.path.exists(RESULTS_PATH):
        existing_df = pd.read_parquet(RESULTS_PATH)
        new_df = pd.concat([existing_df, new_df], ignore_index=True)

    new_df.to_parquet(RESULTS_PATH, index=False)
    print(f"Resultados atualizados em '{RESULTS_PATH}'.")


if __name__ == "__main__":
    main()