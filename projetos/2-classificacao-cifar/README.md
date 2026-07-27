## 📝 Relatório do Candidato

👤 **Nome Completo:** João Victor da Silva Costa Vasconcelos

### 1️⃣ Resumo da Arquitetura do Modelo

Implementei uma CNN para classificar imagens RGB de 32×32 pixels do dataset CIFAR-10. A entrada utiliza data augmentation com espelhamento horizontal, pequenas rotações e zoom aleatório durante o treinamento.

A rede possui três blocos convolucionais com 32, 64 e 128 filtros, respectivamente. Cada bloco é composto por Conv2D com ativação ReLU, BatchNormalization e MaxPooling2D. Após os blocos convolucionais, foram utilizadas as camadas Flatten, Dense com 128 neurônios e ReLU, Dropout de 0,4 e uma camada final Dense com 10 saídas e ativação softmax.

O treinamento utilizou o otimizador Adam, a função de perda sparse categorical crossentropy e EarlyStopping monitorando a perda de validação.

#### Justificativa dos Hiperparâmetros

Foi utilizado `Dropout(0.4)` logo após a camada densa, região que concentra a maior parte dos parâmetros da rede. Durante o treinamento, essa taxa desativa aleatoriamente 40% das ativações, reduzindo a coadaptação entre neurônios e o risco de overfitting. O valor de 0.4 oferece uma regularização relevante, mas ainda mantém 60% das ativações disponíveis em cada atualização, evitando uma perda excessiva de capacidade de aprendizado.

O `batch_size=64` foi escolhido como um equilíbrio entre estabilidade das atualizações, consumo de memória e tempo de treinamento em CPU. Já o `EarlyStopping` com `patience=4` permite tolerar pequenas oscilações na perda de validação, mas interrompe o treinamento quando a melhora deixa de ocorrer, reduzindo treinamento desnecessário e overfitting.

### 2️⃣ Bibliotecas Utilizadas

- Python 3.11.15
- TensorFlow 2.21.0
- Keras 3.12.0
- NumPy 2.2.6

### 3️⃣ Técnica de Otimização do Modelo

Foi utilizada Dynamic Range Quantization durante a conversão do modelo Keras (`model.h5`) para TensorFlow Lite (`model.tflite`), por meio de `tf.lite.Optimize.DEFAULT`.

Essa técnica reduz o tamanho do modelo ao otimizar principalmente os pesos internos, tornando o modelo mais apropriado para dispositivos Edge com limitações de memória e armazenamento. Ela não exigiu um conjunto adicional de imagens para calibração e manteve a interface de entrada e saída em float32.

### 4️⃣ Resultados Obtidos

A acurácia final de validação obtida foi de 75,72%, com perda de validação de 0,7266. O treinamento utilizou EarlyStopping, que interrompeu o processo no epoch 18 de 25 e restaurou os melhores pesos encontrados na validação.

O arquivo `model.h5` ficou com aproximadamente 4,2 MB. Após a conversão e otimização, o arquivo `model.tflite` ficou com aproximadamente 365 KB, uma redução de cerca de 91% no tamanho do modelo.

### 5️⃣ Comentários Adicionais (Opcional)

O modelo foi treinado apenas em CPU e sem o uso de modelos pré-treinados. Este projeto marcou meu primeiro contato prático com Machine Learning, e o principal aprendizado foi compreender o pipeline completo para Edge AI: preparação dos dados, treinamento, validação, salvamento, conversão para TensorFlow Lite e inferência com o modelo otimizado.

O projeto também mostrou que a entrega não termina ao treinar uma CNN: foi necessário converter o modelo e verificar que o artefato `model.tflite` realmente executava inferências. Como limitação, o CIFAR-10 possui imagens pequenas e classes visualmente parecidas, o que torna a classificação mais desafiadora.

### 6️⃣ Exemplo de Inferência

A execução foi realizada com `tf.lite.Interpreter`, carregando especificamente o arquivo `model.tflite`.

```text
Rodando inferência em 5 amostras usando model.tflite:

Amostra 1: predito=cat | real=cat
Amostra 2: predito=ship | real=ship
Amostra 3: predito=ship | real=ship
Amostra 4: predito=airplane | real=airplane
Amostra 5: predito=frog | real=frog
```

Nas cinco amostras testadas, o modelo otimizado acertou todas as classes. Esse resultado não substitui a métrica geral de validação, mas confirma que o arquivo `model.tflite` foi carregado e executado corretamente em inferências individuais.
