# Lê os resultados brutos do benchmark e responde duas perguntas por par de
# modelos, dentro de cada dataset:
#   1) com que frequência a ordem entre eles vira quando troco a semente?
#      (taxa de inversão)
#   2) a diferença média e estatisticamente significativa? (Wilcoxon + Holm)
#
# A lógica toda e por dataset separado: nunca misturo bases diferentes, porque
# o vencedor mudar de uma base pra outra e variaçãoo legítima - o que interessa
# aqui e a variação DENTRO da mesma base, causada só pela semente.

import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

RESULTS_PATH = "resultados.parquet"
METRIC = "accuracy"
ALPHA = 0.05
FIG_DIR = "figs"


def load_results():
    return pd.read_parquet(RESULTS_PATH)


def pairwise_stability(df):
    rows = []

    for ds, group in df.groupby("dataset"):
        # Tabela seeds x modelos: cada linha e uma semente, cada coluna um
        # modelo. dropna remove sementes incompletas (se algum treino falhou).
        wide = group.pivot_table(index="seed", columns="model",
                                 values=METRIC).dropna(axis=0)
        models = list(wide.columns)

        for a, b in itertools.combinations(models, 2):
            va, vb = wide[a].values, wide[b].values
            diff = va - vb                       # diferença pareada, por semente

            win_a = np.mean(diff > 0)            # fraçãoo de sementes em que A venceu
            flip = min(win_a, 1 - win_a)         # taxa de inversao: 0 = estavel, 0.5 = moeda
            mean_diff = diff.mean()

            # Se A e B empatam em toda semente, o Wilcoxon não tem o que testar;
            # trato como p=1 (nenhuma evidencia de diferenca).
            if np.allclose(diff, 0):
                pval = 1.0
            else:
                # Wilcoxon pareado: não supoe normalidade, o que e mais seguro
                # com poucas sementes. zsplit cuida das diferencas iguais a zero.
                pval = stats.wilcoxon(va, vb, zero_method="zsplit").pvalue

            winner = a if mean_diff > 0 else b

            rows.append({
                "dataset": ds,
                "model_a": a,
                "model_b": b,
                "winner_medio": winner,
                "win_rate_vencedor": max(win_a, 1 - win_a),
                "taxa_inversao": flip,
                "dif_media": mean_diff,
                "p_value": pval,
                "n_seeds": len(diff),
            })

    return pd.DataFrame(rows)


def apply_holm_correction(stab_df):
    # Testar dezenas de pares infla os falsos-positivos. Holm corrige isso sem
    # ser tão agressivo quanto Bonferroni. Aplico sobre TODOS os p de uma vez.
    reject, p_adj, _, _ = multipletests(stab_df["p_value"], alpha=ALPHA,
                                        method="holm")
    stab_df = stab_df.copy()
    stab_df["p_holm"] = p_adj
    stab_df["significativo"] = reject
    return stab_df


def generate_report(stab_df):
    total = len(stab_df)
    sig = stab_df["significativo"].sum()

    print(f"Pares analisados: {total}")
    print(f"Diferenças significativas (Holm): {sig} ({100*sig/total:.1f}%)")
    print(f"Não significativas: {100*(total-sig)/total:.1f}%\n")
    print(f"Taxa de inversão média: {stab_df['taxa_inversao'].mean():.3f}")
    print(f"Pares com inversão > 0.40: {(stab_df['taxa_inversao'] > 0.40).sum()}\n")

    # Os 10 pares menos estaveis
    fragil = stab_df.sort_values("taxa_inversao", ascending=False).head(10)
    print("Pares com ranking instável (Top 10):")
    print(fragil[["dataset", "model_a", "model_b", "taxa_inversao",
                  "dif_media", "significativo"]].to_string(index=False))
    print()


def generate_figures(df, stab_df):
    os.makedirs(FIG_DIR, exist_ok=True)

    # Um boxplot por dataset, com os modelos ordenados pela acurácia média.
    for ds, group in df.groupby("dataset"):
        order = (group.groupby("model")[METRIC].mean()
                 .sort_values(ascending=False).index)
        data = [group.loc[group.model == m, METRIC].values for m in order]

        plt.figure(figsize=(7, 4.5))
        plt.boxplot(data, showmeans=True)
        plt.ylabel(METRIC)

        title = (f"{ds}:\ndistribuição por seed" if len(ds) > 20
                 else f"{ds}: distribuição por seed")
        plt.title(title, fontsize=10, pad=10)

        plt.xticks(range(1, len(order) + 1), order, rotation=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{FIG_DIR}/boxplot_{ds}.png", dpi=130)
        plt.close()

    # Histograma geral das taxas de inversão
    plt.figure(figsize=(7, 4))
    plt.hist(stab_df["taxa_inversao"], bins=20, edgecolor="black")
    plt.axvline(0.5, color="red", linestyle="--", label="50% (aleatório)")
    plt.xlabel("Taxa de inversão")
    plt.ylabel("Contagem")
    plt.title("Estabilidade de Ranking entre Modelos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/estabilidade.png", dpi=130)
    plt.close()


def main():
    df = load_results()
    stab = pairwise_stability(df)
    stab = apply_holm_correction(stab)

    stab.to_csv("resumo_ranking.csv", index=False)

    generate_report(stab)
    generate_figures(df, stab)


if __name__ == "__main__":
    main()