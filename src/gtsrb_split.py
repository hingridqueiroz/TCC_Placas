"""
gtsrb_split.py

Divisão do GTSRB sem vazamento de dados:
  - TESTE: conjunto de teste oficial (12.630 imagens), nunca usado no treino.
  - TREINO / VALIDAÇÃO: conjunto de treino oficial (39.209 imagens), dividido
    POR SINAL FÍSICO (grupo), mantendo a proporção das 43 classes (estratificado).

Estrutura esperada da base (download oficial do GTSRB):
  GTSRB/Final_Training/Images/00000/00000_00000.ppm ...   (pastas 00000 a 00042)
  GTSRB/Final_Test/Images/00000.ppm ...
  GTSRB/Final_Test/GT-final_test.csv   (anotações do teste, separador ';')

Nome do arquivo de treino: <sequência>_<quadro>.ppm
  Todos os quadros de uma mesma sequência são o MESMO sinal físico.
  O número da sequência reinicia em cada classe, por isso o grupo é "classe_sequência".

Uso:
  python gtsrb_split.py \
      --treino GTSRB/Final_Training/Images \
      --teste  GTSRB/Final_Test/Images \
      --csv-teste GTSRB/Final_Test/GT-final_test.csv \
      --saida splits
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
N_CLASSES = 43
EXTENSOES = {".ppm", ".png", ".jpg", ".jpeg"}


# ----------------------------------------------------------------------
# 1) Listagem dos arquivos
# ----------------------------------------------------------------------
def listar_treino(raiz_treino):
    """Lê as pastas de classe e identifica o grupo (sinal físico) de cada imagem."""
    linhas = []
    for pasta in sorted(Path(raiz_treino).iterdir()):
        if not (pasta.is_dir() and pasta.name.isdigit()):
            continue
        classe = int(pasta.name)
        for img in sorted(pasta.iterdir()):
            if img.suffix.lower() not in EXTENSOES:
                continue
            sequencia = img.stem.split("_")[0]
            linhas.append(
                {
                    "caminho": str(img),
                    "classe": classe,
                    "grupo": f"{classe}_{sequencia}",
                }
            )
    df = pd.DataFrame(linhas)
    if df.empty:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {raiz_treino}")
    return df


def listar_teste(raiz_teste, csv_gt):
    """Lê o CSV oficial de anotações do teste (colunas Filename e ClassId)."""
    gt = pd.read_csv(csv_gt, sep=";")
    return pd.DataFrame(
        {
            "caminho": [str(Path(raiz_teste) / f) for f in gt["Filename"]],
            "classe": gt["ClassId"].astype(int),
        }
    )


# ----------------------------------------------------------------------
# 2) Divisão treino/validação por grupo
# ----------------------------------------------------------------------
def dividir_treino_val(df, frac_val=0.15, seed=SEED):
    """
    StratifiedGroupKFold:
      - groups: todas as imagens de um mesmo sinal ficam no mesmo conjunto;
      - stratified: proporção de classes aproximadamente preservada.
    Com n_splits = round(1 / frac_val), uma das dobras vira a validação.
    """
    n_splits = round(1 / frac_val)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx_treino, idx_val = next(sgkf.split(df, y=df["classe"], groups=df["grupo"]))
    treino = df.iloc[idx_treino].reset_index(drop=True)
    val = df.iloc[idx_val].reset_index(drop=True)
    return treino, val


def verificar(treino, val, teste):
    """Checagens de sanidade: falham (assert) se houver vazamento ou classe ausente."""
    # 1. Nenhum sinal físico aparece em treino e validação ao mesmo tempo
    inter = set(treino["grupo"]) & set(val["grupo"])
    assert not inter, f"VAZAMENTO: {len(inter)} grupos em treino e validação"

    # 2. Nenhuma imagem repetida entre os conjuntos
    assert not (set(treino["caminho"]) & set(val["caminho"])), "Imagem repetida"
    assert not (set(treino["caminho"]) & set(teste["caminho"])), "Treino no teste"

    # 3. Todas as 43 classes presentes em cada conjunto
    for nome, d in [("treino", treino), ("validação", val), ("teste", teste)]:
        faltando = set(range(N_CLASSES)) - set(d["classe"])
        assert not faltando, f"Classes ausentes em {nome}: {sorted(faltando)}"

    print("Verificações OK: sem vazamento por grupo e com as 43 classes em todos os conjuntos.")
    print(f"  Treino:    {len(treino):6d} imagens | {treino['grupo'].nunique():5d} sinais")
    print(f"  Validação: {len(val):6d} imagens | {val['grupo'].nunique():5d} sinais")
    print(f"  Teste:     {len(teste):6d} imagens (conjunto oficial)")


# ----------------------------------------------------------------------
# 3) Dataset e DataLoaders (mesmo pré-processamento descrito na metodologia)
# ----------------------------------------------------------------------
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]


def transformacoes():
    treino_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEDIA, DESVIO),
        ]
    )
    avaliacao_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEDIA, DESVIO),
        ]
    )
    return treino_tf, avaliacao_tf


class GTSRBDataset(Dataset):
    def __init__(self, df, transform=None):
        self.caminhos = df["caminho"].tolist()
        self.classes = df["classe"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, i):
        img = Image.open(self.caminhos[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.classes[i]


def carregar_dataloaders(pasta_splits, batch_size=32, num_workers=4):
    """Use isto no seu script de treino: lê os CSVs salvos e devolve os 3 loaders."""
    pasta = Path(pasta_splits)
    treino = pd.read_csv(pasta / "treino.csv")
    val = pd.read_csv(pasta / "validacao.csv")
    teste = pd.read_csv(pasta / "teste.csv")
    treino_tf, avaliacao_tf = transformacoes()

    g = torch.Generator().manual_seed(SEED)
    loader_treino = DataLoader(
        GTSRBDataset(treino, treino_tf),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g,
    )
    loader_val = DataLoader(
        GTSRBDataset(val, avaliacao_tf),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    loader_teste = DataLoader(
        GTSRBDataset(teste, avaliacao_tf),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return loader_treino, loader_val, loader_teste


# ----------------------------------------------------------------------
# 4) Execução: gera e salva os CSVs de divisão (reprodutíveis)
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Divisão do GTSRB sem vazamento.")
    ap.add_argument("--treino", required=True, help="pasta Final_Training/Images")
    ap.add_argument("--teste", required=True, help="pasta Final_Test/Images")
    ap.add_argument("--csv-teste", required=True, help="GT-final_test.csv")
    ap.add_argument("--saida", default="splits", help="pasta de saída dos CSVs")
    ap.add_argument("--frac-val", type=float, default=0.15)
    args = ap.parse_args()

    df = listar_treino(args.treino)
    teste = listar_teste(args.teste, args.csv_teste)
    treino, val = dividir_treino_val(df, args.frac_val)

    verificar(treino, val, teste)

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    treino.to_csv(saida / "treino.csv", index=False)
    val.to_csv(saida / "validacao.csv", index=False)
    teste.to_csv(saida / "teste.csv", index=False)
    print(f"CSVs salvos em: {saida.resolve()}")


if __name__ == "__main__":
    main()