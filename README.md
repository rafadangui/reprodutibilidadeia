# Quão Frágil é o Ranking? Estudo Empírico da Estabilidade de Comparações entre Modelos de ML

Este repositório contém o código, as dependências e os resultados do artigo
*"Quão Frágil é o Ranking? Um Estudo Empírico da Estabilidade de Comparações
entre Modelos de Machine Learning"*.

O trabalho investiga, de forma empírica e controlada, **quão estáveis são as
conclusões comparativas entre modelos de aprendizado de máquina** quando se
varia a semente aleatória (que determina a partição treino/teste). Mede-se a
*taxa de inversão de ranking* entre pares de modelos e a significância
estatística das diferenças, por meio de teste de Wilcoxon pareado com correção
de Holm para comparações múltiplas.

## Conteúdo do repositório

| Arquivo | Descrição |
|---|---|
| `benchmark_reprodutibilidade.py` | Executa os treinos (modelo × semente × dataset) e gera `resultados.parquet`. |
| `analise_ranking.py` | Lê `resultados.parquet`, calcula a taxa de inversão e os testes estatísticos (Wilcoxon + Holm) e gera as figuras. |
| `requirements.txt` | Dependências com versões-piso. |
| `requirements-lock.txt` | Versões exatas usadas nos experimentos (reprodutibilidade estrita). |
| `resumo_ranking.csv` | Resultado consolidado: um registro por par de modelos, por dataset. |
| `figs/` | Figuras geradas (boxplots por dataset e histograma de estabilidade). |

## Requisitos

- Python 3.11 ou 3.12
- macOS
- Em macOS com Apple Silicon, instale a biblioteca OpenMP antes (necessária para
  LightGBM e XGBoost):
  ```bash
  brew install libomp
  ```

## Como reproduzir

```bash
# 0. Clonar o repositório
git clone https://github.com/rafadangui/reprodutibilidadeia.git
cd reprodutibilidadeia

# 1. Criar e ativar um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Instalar as dependências
pip install --upgrade pip
pip install -r requirements.txt

# 3. Rodar o experimento (gera resultados.parquet)
python benchmark_reprodutibilidade.py

# 4. Rodar a análise (gera tabelas, resumo_ranking.csv e figuras em figs/)
python analise_ranking.py
```

Para uma execução rápida de teste, reduza o número de datasets e de sementes no
topo de `benchmark_reprodutibilidade.py` (variáveis `DATASET_IDS` e `SEEDS`)
antes de rodar a grade completa.

## Configuração experimental

- **Modelos:** Regressão Logística, Random Forest, XGBoost, LightGBM.
- **Conjuntos de dados:** 10 datasets tabulares públicos da suíte OpenML-CC18.
- **Sementes:** 20 por configuração.
- **Métricas:** acurácia (com intervalo), taxa de inversão de ranking,
  Wilcoxon pareado, correção de Holm.

## Ambiente de referência

Os experimentos foram executados em um Apple M1 Pro (8 núcleos, 16 GB),
com o paralelismo restrito aos 6 núcleos de desempenho. As versões exatas das
bibliotecas estão registradas em `requirements-lock.txt`.

## Citação

Se este material for útil, cite o artigo correspondente. (Adicionar aqui a
referência completa quando publicada.)

## Licença

Distribuído sob a licença MIT. Veja o arquivo `LICENSE`.
