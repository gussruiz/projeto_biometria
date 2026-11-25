# Verificação Facial com ResNet50 e ArcFace — Grupo 2

## 1. Modalidade biométrica e datasets

- **Modalidade:** verificação facial 1:1.
- **Base principal:** CFP-FP (Celebrities in Frontal-Profile), com 500 identidades, 7.000 imagens e 7.000 pares frontal-perfil.
- **Base secundária:** LFW (Labeled Faces in the Wild), versão deep-funneled, com 13.233 imagens de 5.749 indivíduos.

| Base | Etapa | Dados utilizados |
| --- | --- | --- |
| CFP-FP | Treinamento (folds 01–06) | 300 identidades e 4.200 imagens |
| CFP-FP | Validação (folds 07–08) | 100 identidades e 1.400 pares |
| CFP-FP | Teste (folds 09–10) | 100 identidades e 1.400 pares |
| LFW | Teste cross-dataset (DevTest) | 500 pares genuínos e 500 impostores |

A divisão do CFP-FP é mutuamente exclusiva por identidade. Os pares de validação e teste do CFP-FP são balanceados. O LFW é utilizado exclusivamente para avaliação, sem novo treinamento ou ajuste do limiar.

Os datasets não estão incluídos neste repositório. Obtenha-os nos links abaixo:

- [Celebrities in Frontal-Profile in the Wild](http://www.cfpw.io/)
- [Labelled Faces in the Wild](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?resource=download)

## 2. CNN utilizada e origem dos pesos

- **Arquitetura:** ResNet50, disponibilizada pelo `torchvision`.
- **Pesos iniciais:** `ResNet50_Weights.IMAGENET1K_V1`.
- **Cabeça adaptada:** Dropout (0,4), camada linear (2048 → 512) e BatchNorm1d.
- **Parâmetros do extrator:** 24.558.144, dos quais 23.113.216 são ajustáveis.

## 3. Dataset de pré-treinamento

ImageNet-1K, utilizada no pré-treinamento original da ResNet50 para classificação de imagens. O modelo é posteriormente especializado para verificação facial mediante fine-tuning no CFP-FP.

## 4. Pré-processamento e formato de entrada

1. Leitura da imagem com OpenCV.
2. Detecção da maior face com YuNet e localização de cinco landmarks faciais.
3. Alinhamento facial por transformação afim parcial, recorte e redimensionamento para 112 × 112 pixels.
4. Recorte central quando a detecção ou o alinhamento falham.
5. Conversão para RGB e normalização com média `[0.485, 0.456, 0.406]` e desvio-padrão `[0.229, 0.224, 0.225]` da ImageNet.

**Formato de entrada:** tensor `[N, 3, 112, 112]` (RGB).

Durante o treinamento, são aplicados espelhamento horizontal aleatório, ColorJitter (0,2) e rotações de até ±10°. Validação e teste não recebem transformações aleatórias.

## 5. Dimensão do embedding

Cada imagem é representada por um vetor de **512 dimensões**, normalizado pela norma L2 durante a extração de embeddings para avaliação.

## 6. Função de perda e estratégia de treinamento/fine-tuning

Utiliza-se ArcFace (`ArcMarginProduct`) associado à `CrossEntropyLoss`, com margem angular aditiva para favorecer a compactação intraclasse e a separação interclasse.

As camadas `conv1`, `bn1`, `layer1` e `layer2` são congeladas. As camadas `layer3`, `layer4`, a nova cabeça de embeddings e a camada ArcFace são treinadas. O classificador ArcFace é empregado somente durante o treinamento.

| Hiperparâmetro | Valor |
| --- | --- |
| Escala ArcFace | 30 |
| Margem angular | 0,5 radiano |
| Otimizador | Adam |
| Learning rate do backbone ajustável | 0,0001 |
| Learning rate da cabeça e ArcFace | 0,001 |
| Weight decay | 0,0005 |
| Scheduler | CosineAnnealingLR (T_max=20) |
| Batch size (treino) | 32 |
| Batch size (extração) | 64 |
| Máximo de épocas | 20 |
| Early stopping | 4 épocas sem melhora |
| Seleção do checkpoint | Menor EER de validação |

## 7. Método de matching e regra de threshold

O matching é realizado por similaridade de cosseno entre embeddings normalizados:

```text
score(a, b) = (a · b) / (||a||₂ × ||b||₂)
```

O limiar é definido exclusivamente na validação do CFP-FP, no ponto em que a diferença absoluta entre FMR e FNMR é mínima, e permanece fixo nos testes CFP-FP e LFW, na robustez e na ablação.

**Threshold principal (seed 42):** `0.13629400730133057`.

```text
score >= threshold → aceitar como genuíno
score <  threshold → rejeitar como impostor
```

Cada semente adicional possui seu próprio limiar, obtido pela mesma regra na validação.

## 8. Versão do Python e bibliotecas

| Componente | Versão registrada |
| --- | --- |
| Python | 3.12.10 |
| PyTorch | 2.5.1+cu121 |
| CUDA (build do PyTorch) | 12.1 |
| tqdm | 4.67.1 |

Demais bibliotecas utilizadas: `torchvision`, `numpy`, `pandas`, `opencv-python-headless`, `Pillow`, `scikit-learn` e `matplotlib`. As versões exatas dessas bibliotecas não foram registradas nos materiais fornecidos.

## 9. Hardware utilizado

- **GPU:** NVIDIA GeForce RTX 3050 Ti Laptop GPU.
- **Dispositivo de execução:** CUDA.

As especificações de CPU, RAM e driver não foram registradas.

## 10. Seeds utilizadas

As sementes utilizadas foram **42, 123 e 2026**, com a seed 42 como execução principal. São configurados `random`, NumPy, PyTorch e CUDA, mantendo a divisão por identidade e os pares de avaliação fixos.

## 11. Comandos para executar o pipeline e reproduzir os resultados do artigo

### Preparar o ambiente

Na raiz do repositório, com Python 3.12 instalado:

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Instale as dependências (ambiente sugerido compatível com CUDA 12.1; não corresponde a um registro integral do ambiente original):

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy pandas opencv-python-headless Pillow scikit-learn matplotlib tqdm==4.67.1 jupyterlab ipykernel ipywidgets nbconvert
python -m ipykernel install --user --name biometria-cfp --display-name "Python (Biometria CFP)"
```

### Organizar os datasets

Obtenha os datasets pelos links da seção 1, respeitando seus termos de uso.

```text
 data/
 ├── cfp-dataset/
 │   ├── Data/Images/
 │   └── Protocol/Split/
 │       ├── Pair_list_F.txt
 │       ├── Pair_list_P.txt
 │       └── FP/ (folds 01 a 10, same.txt e diff.txt)
 └── lfw/
     ├── matchpairsDevTest.csv
     ├── mismatchpairsDevTest.csv
     └── lfw-deepfunneled/lfw-deepfunneled/
```

### Executar o notebook

```bash
jupyter lab
```

Abra `Projeto_Biometria_Didatico_v2(2).ipynb`, selecione o kernel **Python (Biometria CFP)** e ajuste os caminhos na Parte 0 conforme a localização dos datasets. Se o notebook tiver outro nome no repositório, utilize o nome correspondente.

```python
DATA_ROOT = Path.cwd() / "data"
WORK_DIR = Path.cwd() / "saidas"
CFP_RAW = DATA_ROOT / "cfp-dataset"
LFW_RAW = DATA_ROOT / "lfw" / "lfw-deepfunneled" / "lfw-deepfunneled"
LFW_CSV = DATA_ROOT
SEED = 42
RETRAIN = False
```

Execute todas as células em ordem (Run All Cells). Alternativamente, execute pelo terminal após configurar os caminhos:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=biometria-cfp --ExecutePreprocessor.timeout=-1 --output=Projeto_Biometria_Executado.ipynb "Projeto_Biometria_Didatico_v2(2).ipynb"
```

Para reutilizar os pesos originais, mantenha `RETRAIN = False` e coloque `best_seed42.pt`, `best_seed123.pt` e `best_seed2026.pt` em `saidas/checkpoints/`. Se não houver checkpoints, o notebook realiza o treinamento. Para refazer integralmente os experimentos, use uma pasta `WORK_DIR` nova e vazia e configure `RETRAIN = True`.

A primeira execução requer internet para baixar os pesos iniciais e o detector YuNet. Os resultados são gravados em `saidas/results/`. Os checkpoints originais não estavam no ZIP fornecido; a reprodução exata das métricas depende da disponibilidade desses pesos e do ambiente original.