## Explicando as previsões de um modelo com LIME: um exemplo de cálculo de importância de características

- Projeto: Análise de importância de características em modelo de classificação com LIME



---

**Objetivo**: Desenvolver um modelo de machine learning que classifica se um paciente tem ou não uma determinada doença, com base em suas características, como idade, gênero, histórico médico, resultados de testes etc. Queremos usar o LIME para explicar por que o modelo classificou um determinado paciente como tendo a doença.

---

### Mas, afinal, o que é o LIME?

LIME significa *Local Interpretable Model-agnostic Explanations* e é uma técnica que pode ser usada para explicar como um modelo de machine learning tomou suas decisões. Ele não se concentra em calcular a importância relativa de cada recurso, mas em explicar como cada recurso influenciou a previsão do modelo para uma observação específica.

```python
import lime
import lime.lime_tabular
import numpy as np
```

> Nesse exemplo, criamos um modelo simples que classifica os pacientes com base em sua idade, e criamos uma nova amostra com a idade alterada em 2 anos. Em seguida, usamos o LIME para explicar a importância da idade na previsão do modelo, calculando a derivada parcial da previsão do modelo em relação à idade.


```python
# Criando um modelo de exemplo
def predict_fn(x):
    return np.column_stack([(x[:, 0] > 50).astype(int), 1 - (x[:, 0] > 50).astype(int)])

# Criando amostras de dados de exemplo
data = np.random.randint(0, 100, size=(100, 5))

# Especificando as características
feature_names = ['idade', 'sexo', 'pressão arterial', 'glicemia', 'colesterol']

# Criando um objeto explicador do LIME
explainer = lime.lime_tabular.LimeTabularExplainer(data, feature_names=feature_names, class_names=['Não', 'Sim'], discretize_continuous=False)

# Criando uma nova amostra com idade alterada em 2 anos
new_data = data.copy()
new_data[:, 0] += 2

# Obtendo as previsões para as amostras originais e a nova amostra
original_preds = predict_fn(data)
new_preds = predict_fn(new_data)
```

> A linha exp = explainer.explain_instance(new_data[0], predict_fn, num_features=5) calcula a importância das cinco características mais relevantes na previsão do modelo para a nova amostra. A linha age_importance = exp.as_list()[0][1] obtém a importância da idade a partir da lista de importâncias retornada pelo LIME.


```python
# Calculando a importância da idade usando o LIME
exp = explainer.explain_instance(new_data[0], predict_fn, num_features=5)

# Obtendo a importância da idade
age_importance = exp.as_list()[0][1]

exp.as_list()
```




    [('idade', -0.46546278875695796),
     ('sexo', 0.0136130475168977),
     ('glicemia', -0.005174571831061595),
     ('colesterol', 0.001579964371325525),
     ('pressão arterial', 0.00047667378765386633)]




```python
# Imprimindo a importância da idade
print('A importância da idade é:', age_importance)
```

    A importância da idade é: -0.46388716808645064
    

> O valor de -0.46546278875695796 é a derivada parcial da previsão do modelo em relação à idade. Tal resultado mostra a importância da idade para a previsão do modelo. Esse valor pode ser usado para identificar as características mais importantes para o modelo, ajudando a entender como o modelo toma suas decisões.
