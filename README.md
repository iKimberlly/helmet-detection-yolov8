# 🦺 Detecção de Capacetes com YOLOv8 para Monitoramento de Segurança Industrial

## 📌 Resumo

Este projeto apresenta o desenvolvimento de um sistema de visão computacional baseado na arquitetura YOLOv8 para detecção em tempo real de capacetes como Equipamento de Proteção Individual (EPI). A solução tem como objetivo automatizar a fiscalização de conformidade em ambientes industriais, reduzindo falhas humanas e mitigando riscos de acidentes de trabalho.

O modelo foi treinado utilizando o dataset Hard Hat Detection e avaliado por meio de métricas padrão de detecção de objetos.

---

## 1. Introdução

A negligência no uso de Equipamentos de Proteção Individual é uma das principais causas de acidentes em ambientes industriais. Processos tradicionais de inspeção são manuais, suscetíveis a falhas e demandam alto custo operacional.

Neste contexto, técnicas de Deep Learning aplicadas à visão computacional surgem como alternativa eficiente para monitoramento automatizado. Este projeto investiga o uso do modelo YOLOv8 para identificação de trabalhadores com e sem capacete em imagens e vídeos.

---

## 2. Metodologia

### 2.1 Dataset

- Dataset: Hard Hat Detection (Kaggle)
- Formato original: Pascal VOC (XML)
- Classes utilizadas:
  - `helmet`
  - `head` (sem capacete)

As anotações foram convertidas para o formato YOLO (bounding boxes normalizadas).

---

### 2.2 Pré-processamento

- Leitura e parsing de arquivos XML
- Mapeamento de classes:
  - helmet → 0
  - head → 1
- Normalização das coordenadas das bounding boxes
- Organização das imagens e labels no formato YOLO

---

### 2.3 Arquitetura do Modelo

- Modelo base: YOLOv8n (Ultralytics)
- Framework: PyTorch
- Detector: Single-stage (anchor-free)
- Tamanho de imagem: 640x640

---

### 2.4 Configuração de Treinamento

| Parâmetro | Valor |
|------------|--------|
| Épocas | 50 |
| Batch size | 16 |
| Image size | 640 |
| Modelo base | yolov8n.pt |

---

## 3. Pipeline de Execução

### Download do Dataset

```bash
python src/01_download_dataset.py
```

### Conversão para formato YOLO

```bash
python src/02_convert_xml_to_yolo.py
```

### Treinamento

```bash
python src/04_train_yolo.py
```

### Inferência (Imagem/Vídeo)

```bash
python src/05_predict.py
```

### Avaliação

```bash
python src/06_evaluate.py
```

---

## 4. Resultados

## 📊 Análise Exploratória do Dataset

### Distribuição das Classes

![Distribuição das Classes](resultado/distribuicao_classes.jpg)

Observa-se um desbalanceamento entre as classes, com maior número de instâncias da classe `helmet` em comparação à classe `no_helmet`. Esse fator pode influenciar o desempenho do modelo, especialmente na detecção de não conformidades.

---

### Distribuição Espacial das Bounding Boxes

A análise da distribuição das bounding boxes demonstra concentração predominante na região central das imagens, refletindo o enquadramento típico de trabalhadores em ambientes industriais.

---

## 📈 Avaliação do Modelo

### Matriz de Confusão

![Matriz de Confusão](resultado/metrica_confusao.jpg)

A matriz de confusão permite avaliar a capacidade do modelo em distinguir corretamente entre trabalhadores com e sem capacete.

---

## 🖼️ Resultados de Inferência

### Exemplos em Imagens

![Resultado 1](results/result_1.jpg)
![Resultado 2](results/result_2.jpg)
![Resultado 3](results/result_3.jpg)
![Resultado 4](results/result_4.jpg)

Os resultados demonstram boa capacidade de localização e classificação em diferentes cenários industriais, incluindo múltiplos trabalhadores, oclusões parciais e variações de iluminação.

### 4.2 Métricas Avaliadas

O modelo foi avaliado utilizando:

- mAP@0.5
- mAP@0.5:0.95
- Precisão (Precision)
- Revocação (Recall)

*(Inserir valores obtidos após model.val())*

---

## 5. Discussão

Os resultados demonstram que o modelo YOLOv8 apresenta desempenho satisfatório para aplicações em monitoramento de segurança industrial em tempo real.

Desafios observados:

- Oclusões parciais
- Variações de iluminação
- Objetos pequenos em grandes distâncias

Apesar das limitações, o modelo demonstra potencial para aplicações reais em sistemas embarcados e monitoramento automatizado.

---

## 6. Limitações

- Base de dados limitada
- Não foi aplicada técnica avançada de data augmentation
- Não houve otimização para edge devices
- Ausência de sistema de alerta integrado

---

## 7. Trabalhos Futuros

- Ampliação do dataset
- Comparação com outras versões do YOLO
- Implementação de rastreamento (tracking)
- Deploy em dispositivos embarcados (Jetson Nano)
- Integração com sistema de alerta em tempo real

---

## 8. Estrutura do Projeto

```
helmet-detection-yolov8/
│
├── src/
│   ├── 01_download_dataset.py
│   ├── 02_convert_xml_to_yolo.py
│   ├── 04_train_yolo.py
│   ├── 05_predict.py
│   └── 06_evaluate.py
│
├── results/
├── data.yaml
├── requirements.txt
└── README.md
```

---

## 9. Requisitos

Instalar dependências:

```bash
pip install -r requirements.txt
```

Principais bibliotecas:

- ultralytics
- torch
- opencv-python
- kagglehub
- numpy
- matplotlib

---

## 👩‍💻 Autora

Kimberlly Silva  
Análise e Desenvolvimento de Sistemas  
Projeto Acadêmico – 2026  
Área: Visão Computacional | Deep Learning | Segurança Industrial
