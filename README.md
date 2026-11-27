# Classificação de Placas de Trânsito com Transfer Learning (MobileNetV2)

Projeto de TCC (Trabalho de Conclusão de Curso) que treina e avalia um classificador de
placas de trânsito (sinais de trânsito) usando **Transfer Learning** com **MobileNetV2**
no PyTorch, além de um pipeline completo de **fine-tuning independente** por blocos da
backbone.

O dataset utilizado é o **GTSRB (German Traffic Sign Recognition Benchmark)**, com
**43 classes** de placas e mais de **50.000 imagens** no total.

> **Nota:** Os resultados experimentais (pesos de modelos treinados, logs, métricas
> geradas e o próprio dataset) **não estão incluídos** neste repositório por motivos de
> tamanho e de licenciamento do dataset. Apenas o código é publicado. Veja a seção
> [Reprodução dos experimentos](#reprodução-dos-experimentos).

---

## Sumário

- [Visão geral](#visão-geral)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Arquitetura](#arquitetura)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Preparação do dataset](#preparação-do-dataset)
- [Uso](#uso)
- [Avaliação](#avaliação)
- [Reprodução dos experimentos](#reprodução-dos-experimentos)
- [Licença](#licença)

---

## Visão geral

O projeto explora duas fases clássicas de Transfer Learning:

1. **Feature Extraction (FE):** backbone MobileNetV2 congelada, treinando apenas a
   cabeça classificadora. Gera o modelo base (`models/fe_model.pth`) reutilizado por
   todos os experimentos de fine-tuning.
2. **Fine-Tuning (FT) independente:** cada experimento parte **do mesmo modelo FE**
   (sem acúmulo de pesos entre experimentos), descongelando diferentes subconjuntos de
   blocos da backbone (`model.features`, índices 0–18) para comparar o impacto de
   quais camadas são descongeladas na acurácia e no tempo de treino.

A divisão do dataset é feita de forma a **evitar vazamento de dados**: o conjunto de
teste é o oficial do GTSRB (nunca usado no treino) e a divisão treino/validação é
feita **por sinal físico** (grupo), mantendo a proporção das 43 classes (estratificada).

---

## Estrutura do projeto

```	ext
TCC_Placas/
├── src/
│   ├── config.py           # Constantes compartilhadas (classes, seeds, hiperparâmetros)
│   ├── dataset.py          # DataLoaders (treino/val/teste) via CSVs em data/splits
│   ├── model.py            # MobileNetV2 + cabeça customizada e helpers de freeze/unfreeze
│   ├── utils.py            # Seeds, device, TimeTracker, save/load, métricas, treino/val
│   ├── train_fe.py         # Feature Extraction (backbone congelada)
│   ├── train_ft.py         # Fine-tuning independente por blocos
│   ├── test.py             # Avaliação (métricas, matriz de confusão, exemplos, erros)
│   └── gtsrb_split.py      # Divisão do GTSRB sem vazamento (teste oficial + grupos)
├── classificador_placas.ipynb  # Notebook exploratório (Colab) do pipeline inicial
├── requirements.txt
├── .gitignore
└── README.md

```

Pastas geradas em tempo de execução (não versionadas):

```
data/            # Dataset (original baixado e/ou splits gerados)
models/          # Pesos do modelo FE (fe_model.pth) e metadados
resultados_mobilenetv2/  # Resultados dos experimentos de FT
results_test/    # Métricas, matrizes de confusão, exemplos, erros
logs/            # Logs de treino (FE e FT) e de teste
```

---


## Arquitetura

- **Backbone:** `torchvision.models.mobilenet_v2` pré-treinada no ImageNet.
- **Cabeça classificadora:** `Dropout(0.2) → Linear(1280, 512) → ReLU → Dropout(0.2) → Linear(512, 43)`.
- **Entrada:** RGB `224×224`, normalização ImageNet (`mean=[0.485, 0.456, 0.406]`,
  `std=[0.229, 0.224, 0.225]`).
- **Transforms de treino:** `Resize(256) → RandomResizedCrop(224)` + normalização.
- **Transforms de val/teste:** `Resize(256) → CenterCrop(224)` + normalização.
- **Otimizador:** Adam com learning rates diferenciadas (backbone `1e-5`, classificador
  `1e-4` no FT; `1e-3` no FE). Scheduler `ReduceLROnPlateau` (`factor=0.5`, `patience=2`).
- **Early stopping:** paciência de 5 épocas (FT).
- **Reprodutibilidade:** seed fixa em `random`, `numpy` e `torch` (42 por padrão).

A backbone MobileNetV2 possui 19 módulos em `model.features` (índices 0–18): o índice 0
é a primeira convolução, os índices 1–17 são blocos invertidos residuais e o índice 18 é
a convolução final. O fine-tuning parcial seleciona subconjuntos dos índices 1–17; o
full fine-tuning descongela 0–18.

---

## Pré-requisitos

- Python 3.10+ (testado em 3.11)
- PyTorch com suporte a CUDA (recomendado) ou CPU
- ~5 GB de espaço para o dataset GTSRB

---

## Instalação

```bash
# Clone o repositório
git clone https://github.com/hingridqueiroz/TCC_Placas.git
cd TCC_Placas

# Crie e ative um ambiente virtual
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

Para PyTorch com CUDA, instale antes o wheel adequado ao seu driver (ex.:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).

---

## Preparação do dataset

### Divisão recomendada (sem vazamento)

1. Baixe o GTSRB oficial. Você pode baixar diretamente via `torchvision` (recomendado):

```python
from torchvision.datasets import GTSRB

# Baixa o conjunto de treino e teste oficiais
train_set = GTSRB(root="./GTSRB", split="train", download=True)
test_set  = GTSRB(root="./GTSRB", split="test",  download=True)
```

Isso cria a estrutura esperada em `./GTSRB/`:

```text
GTSRB/
├── GTSRB/Training/Images/00000/00000_00000.ppm ...
└── GTSRB/Test/Images/00000.ppm ...
```

> Alternativamente, baixe manualmente do site oficial do GTSRB (ou via Kaggle) e
> organize conforme a estrutura acima.

2. Gere a divisão sem vazamento (teste oficial + treino/val por sinal físico):

```bash
python src/gtsrb_split.py \
  --treino GTSRB/GTSRB/Training/Images \
  --teste  GTSRB/GTSRB/Test/Images \
  --csv-teste GTSRB/GTSRB/Test/GT-final_test.csv \
  --saida data/splits
```

Opções: `--saida` (padrão `splits`) e `--frac-val` (padrão `0.15`).

O script gera `data/splits/{treino,validacao,teste}.csv` (colunas `caminho`, `classe`
e, em treino/validação, `grupo`) e executa checagens de sanidade (sem vazamento por
grupo, sem imagem repetida, todas as 43 classes em cada conjunto).

> Execute o script a partir da raiz do projeto (ou passe caminhos absolutos). O
> `Dataset` resolve caminhos relativos a partir da raiz do projeto, então os CSVs devem
> ser gerados nesse contexto.



## Uso

Os scripts de treino/avaliação usam por padrão `data/splits` (configurável via
`--data-dir`).

### 1. Feature Extraction (modelo base)

```bash
python src/train_fe.py --epochs 40 --lr 1e-3 --batch-size 32 --seed 42
```

Gera `models/fe_model.pth`, `models/fe_time.json` e `models/fe_metadata.json`
(o caminho do modelo pode ser alterado com `--model-save`).

### 2. Fine-Tuning independente

Cada execução parte do mesmo `fe_model.pth` (sem transferência de pesos entre
experimentos). Escolha **apenas um** modo por execução (mutuamente exclusivos):

```bash
# Full fine-tuning (descongela todo o backbone)
python src/train_ft.py --full

# Últimos N blocos
python src/train_ft.py --last-n 6

# Primeiros N blocos
python src/train_ft.py --first-n 6

# Blocos específicos
python src/train_ft.py --blocks 2 5 8 11 14 17
```

Opções úteis: `--epochs`, `--lr-backbone`, `--lr-classifier`, `--batch-size`,
`--seed`, `--fe-model`, `--data-dir`.

Os resultados são salvos em `results_train/ft_independent/<experimento>/`
(`model.pth`, `metadata.json`, `history.json`, `time.json`) e o log em
`logs/independent/<experimento>/training.log`. Ajuste `RESULTS_BASE_DIR` e
`LOGS_BASE_DIR` em `src/train_ft.py` para gravar em outro local.

### 3. Avaliação

```bash
python src/test.py --model resultados_mobilenetv2/independent_full/model.pth \
  --save-examples --num-examples 5
```

Opções: `--data-dir`, `--batch-size`, `--seed`, `--output-dir`, `--run-name`,
`--save-examples`, `--num-examples`.

Gera, em `results_test/<modelo>/run_N/`: `metrics.json`, `confusion_matrix.png`,
`classification_report.txt`, `training_curves.png`, `prediction_examples.png`,
`wrong_predictions/` (imagens e resumo dos erros), `test_time.txt` e `run_info.json`
(que registra o `data_dir` e a data da execução). Cada execução recebe uma pasta nova
(`run_1`, `run_2`, ...) para não sobrescrever resultados anteriores.

---

## Avaliação

As métricas calculadas (via `sklearn`) são: acurácia, precisão/recall/F1 (macro e
weighted), relatório de classificação por classe, matriz de confusão e análise de
predições incorretas (imagens salvas por classe real/predita).

---

## Reprodução dos experimentos

1. Prepare o dataset (seção [Preparação do dataset](#preparação-do-dataset)).
2. Treine o FE: `python src/train_fe.py`.
3. Rode os FT desejados (seção [Fine-Tuning independente](#2-fine-tuning-independente)).
4. Avalie cada modelo:
   `python src/test.py --model resultados_mobilenetv2/<experimento>/model.pth --save-examples`.
5. Compare acurácia de validação, parâmetros treináveis e tempo de treino em
   `resultados_mobilenetv2/<experimento>/metadata.json` (`best_val_acc`,
   `trainable_params`, `exp_time`).

Para reprodutibilidade, todos os scripts aceitam `--seed` (padrão 42).

---

## Licença

Este projeto é distribuído para fins acadêmicos. O dataset GTSRB possui seus próprios
termos de uso; consulte a fonte original antes de redistribuir os dados.
